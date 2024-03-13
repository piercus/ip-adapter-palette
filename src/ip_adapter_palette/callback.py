from pathlib import Path
from refiners.training_utils.common import (
    count_learnable_parameters,
    human_readable_number,
    compute_grad_norm
)
from refiners.fluxion.utils import save_to_safetensors
from refiners.training_utils.callback import Callback, CallbackConfig
from loguru import logger
from torch import norm, nn
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ip_adapter_palette.trainer import PaletteTrainer

from torch.nn.modules.module import Module as TorchModule

class OffloadToCPUConfig(CallbackConfig):
    use: bool = False
class TimestepLossRescalerConfig(CallbackConfig):
    use: bool = False

class MonitorGradientConfig(CallbackConfig):
    patterns: list[str] = []
    total: bool = True
    cross_attention_ratios: bool = True

class OffloadToCPU(Callback[Any]):
    def __init__(self, config: OffloadToCPUConfig) -> None:
        self.config = config
        super().__init__()
    def on_evaluate_begin(self, trainer: "PaletteTrainer") -> None:
        if self.config.use:
            trainer.sd.lda.to(trainer.device)
            trainer.text_encoder.to(trainer.device)
            trainer.image_encoder.to(trainer.device)

    def on_evaluation_end(self, trainer: "PaletteTrainer") -> None:
        if self.config.use:
            trainer.sd.lda.to("cpu")
            trainer.text_encoder.to("cpu")
            trainer.image_encoder.to("cpu")

    # def on_train_begin(self, trainer: "PaletteTrainer") -> None:
    #     trainer.ip_adapter.inject()

class LogModelParamConfig(CallbackConfig):
    use: bool = True

class MonitorTimeConfig(CallbackConfig):
    use: bool = True

class LogModelParam(Callback[Any]):
    def on_init_end(self, trainer: "PaletteTrainer") -> None:
        for key in trainer.models.keys():
            logger.info(f"{key} : {human_readable_number(number=count_learnable_parameters(trainer.models[key].learnable_parameters))}")

class SaveBestModelConfig(CallbackConfig):
    max_checkpoints: int = 3
    overwrite: bool = False

class SaveBestModel(Callback[Any]):
    current_epoch_losses: list[float] = []

    def __init__(self, config: SaveBestModelConfig) -> None:
        self.best_loss = float("inf")
        self.max_checkpoints = config.max_checkpoints
        self.overwrite = config.overwrite

    
    def on_init_end(self, trainer: "PaletteTrainer") -> None:
        self.folder = "./tmp/weights/"+trainer.config.wandb.name
        # create the local folder
        if not self.overwrite:
            Path(self.folder).mkdir(parents=True, exist_ok=False)
        else:
            if Path(self.folder).exists():
                logger.warning(f"Overwriting the folder {self.folder}")
            Path(self.folder).mkdir(parents=True, exist_ok=True)

    def on_compute_loss_end(self, trainer: "PaletteTrainer") -> None:
        if not isinstance(trainer.loss, list):
            raise ValueError("Expected loss to be a list of tensor")
        self.current_epoch_losses.append(sum([loss.item() for loss in trainer.loss]))

    def on_epoch_start(self, trainer: "PaletteTrainer") -> None:
        self.current_epoch_losses = []

    def on_epoch_end(self, trainer: "PaletteTrainer") -> None:
        loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
        
        if loss < self.best_loss:
            self.best_loss = loss
            save_to_safetensors(
                 f"{self.folder}/best_model_{loss:.4f}.safetensors", trainer.ip_adapter.state_dict()
            )

        models = sorted(
            Path(self.folder).glob("best_model_*.safetensors"),
            key=lambda x: float(x.stem.split("_")[-1]),
        )
        print(f"Number of models: {len(models)}")
        if len(models) > self.max_checkpoints:
            worst_model = models[-1]
            worst_model.unlink()

import time

# Ported from open-muse
class AverageTimeMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
    
    def start(self):
        self.start_time = time.time()
    
    def cancel(self):
        self.start_time = None
    
    def stop(self):
        if self.start_time is None:
            return
        spent = time.time() - self.start_time
        self.update(spent)
        self.start_time = None


class MonitorTime(Callback[Any]):
    def __init__(self, config: MonitorTimeConfig) -> None:
        self.forward = AverageTimeMeter()
        self.data = AverageTimeMeter()
        self.backward = AverageTimeMeter()
        
        super().__init__()
    
    def on_compute_loss_begin(self, trainer: "PaletteTrainer") -> None:
        self.forward.start()
    
    def on_compute_loss_end(self, trainer: "PaletteTrainer") -> None:
        self.forward.stop()
    
    def on_backward_begin(self, trainer: "PaletteTrainer") -> None:
        self.backward.start()
    
    def on_backward_end(self, trainer: "PaletteTrainer") -> None:
        self.backward.stop()
    
    def on_batch_begin(self, trainer: "PaletteTrainer") -> None:
        self.data.stop()
    
    def on_batch_end(self, trainer: "PaletteTrainer") -> None:
        trainer.wandb_log(data={
            "timing/forward_time per item": self.forward.avg / trainer.clock.batch_size, 
            "timing/backward_time per item": self.backward.avg / trainer.clock.batch_size, 
            "timing/data_time per item": self.data.avg / trainer.clock.batch_size
        })
        self.data.start()
       
    def on_epoch_begin(self, trainer: "PaletteTrainer") -> None:
        self.data.start()
        
    def on_epoch_end(self, trainer: "PaletteTrainer") -> None:
        self.data.cancel()

from fnmatch import fnmatch

class MonitorGradient(Callback[Any]):
    def __init__(self, config: MonitorGradientConfig) -> None:
        self.config = config
        self.cross_attention_ratios = config.cross_attention_ratios
        super().__init__()
    
    def per_layer_learnable_parameters(self, models: dict[str, TorchModule]) -> dict[str, nn.Parameter]:
        result : dict[str, nn.Parameter] = {}
        for key in models.keys():
            named_parameters = models[key].model.named_parameters()
            result.update({
                f"{key}.{name}": param
                for name, param in named_parameters
            })
        return result

    def on_optimizer_step_begin(self, trainer: "PaletteTrainer") -> None:
        layer_learnable_parameters = self.per_layer_learnable_parameters(trainer.models) # type: ignore

        layer_norms: dict[str, float] = {}

        for layer_name in layer_learnable_parameters:
            param = layer_learnable_parameters[layer_name]
            for pattern in self.config.patterns:
                if fnmatch(layer_name, pattern) and param.grad is not None:
                    norm = compute_grad_norm([param])
                    layer_norms[layer_name] = norm
                    trainer.wandb_log(data={f"layer_grad_norm/{layer_name}": norm})
        
        if self.config.total:
            trainer.wandb_log(data={"grad_norm": trainer.total_gradient_norm})

        eps = 1e-8

        if self.cross_attention_ratios:
            cross_attention_layer_info = [{
                "prefix" : name.split("CrossAttentionBlock")[0],
                "type": "key" if name.endswith("Chain_1.Linear.weight") else "value",
                "norm": layer_norms[name]
            } for name in layer_norms if "CrossAttentionBlock" in name]
            # group by prefix
            prefixes = set([layer["prefix"] for layer in cross_attention_layer_info])
            cross_attention_infos : dict[str, float] = {}
            for prefix in prefixes:
                layers = [layer for layer in cross_attention_layer_info if layer["prefix"] == prefix]
                if len(layers) != 2:
                    raise ValueError(f"Expected 2 layers for prefix {prefix}, got {len(layers)}")
                if layers[0]["type"] == layers[1]["type"]:
                    raise ValueError(f"Expected 2 layers with different types for prefix {prefix}, got {layers[0]['type']} and {layers[1]['type']}")
                
                key_layer = layers[0] if layers[0]["type"] == "key" else layers[1]
                value_layer = layers[0] if layers[0]["type"] == "value" else layers[1]
                cross_attention_infos[prefix] = key_layer["norm"]/(value_layer["norm"] + eps)

            for name in cross_attention_infos:
                trainer.wandb_log(data={f"cross_attention_ratios/{name}": cross_attention_infos[name]})


import math 
from torch import Tensor, exp
class TimestepLossRescaler(Callback[Any]):
    def __init__(self, config: TimestepLossRescalerConfig) -> None:
        self.config = config
        super().__init__()
    
    @staticmethod
    def approximate_loss(inverse_timestep: Tensor, /) -> Tensor:
        a = 3.1198626909458634e-08
        exponent = 2.3683577564059
        b = -0.3560275587290773
        c = -13.269541143845919
        C = 0.36245161978354973
        return a * inverse_timestep**exponent + b * exp(-c / (inverse_timestep - 1001)) + C

    def on_compute_loss_end(self, trainer: "PaletteTrainer") -> None:
        if self.config.use:
            inverse_timestep = 999 - trainer.timestep
            if not isinstance(trainer.loss, list):
                raise ValueError("Expected loss to be a list of tensor")
            loss = trainer.loss[0]
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) / self.approximate_loss(inverse_timestep)
            loss = loss.mean()
            trainer.loss[0] = loss
        else:
            trainer.loss[0] = trainer.loss[0].mean()

# class MmdEvaluation(Callback[Any]):
#     def eval_dataset(self, trainer: "PaletteTrainer") -> None:
#         pass
#     def on_evaluate_begin(self, trainer: "PaletteTrainer") -> None:
#         trainer.eval_da
#         trainer.wandb_log(data={"mmd": trainer.mmd.item()})
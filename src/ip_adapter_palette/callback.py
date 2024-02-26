from pathlib import Path
from refiners.training_utils.common import (
    count_learnable_parameters,
    human_readable_number,
    compute_grad_norm,
)
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

class OffloadToCPU(Callback[Any]):
    def __init__(self, config: OffloadToCPUConfig) -> None:
        self.config = config
        super().__init__()
    def on_evaluation_start(self, trainer: "PaletteTrainer") -> None:
        if self.config.use:
            trainer.sd.lda.to(trainer.device)
            trainer.text_encoder.to(trainer.device)
    
    def on_evaluation_end(self, trainer: "PaletteTrainer") -> None:
        if self.config.use:
            trainer.sd.lda.to("cpu")
            trainer.text_encoder.to("cpu")

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
    max_checkpoints: int = 5


class SaveBestModel(Callback[Any]):
    current_epoch_losses: list[float] = []

    def __init__(self, config: SaveBestModelConfig) -> None:
        self.best_loss = float("inf")
        self.max_checkpoints = config.max_checkpoints

    def on_compute_loss_end(self, trainer: "PaletteTrainer") -> None:
        self.current_epoch_losses.append(trainer.loss.item())

    def on_epoch_start(self, trainer: "PaletteTrainer") -> None:
        self.current_epoch_losses = []

    def on_epoch_end(self, trainer: "PaletteTrainer") -> None:
        loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
        if loss < self.best_loss:
            self.best_loss = loss
            # TODO: save adapter
            # save_to_safetensors(
            #     f"best_model_{loss:.4f}.safetensors", trainer.auto_encoder.state_dict()
            # )

        models = sorted(
            Path(".").glob("best_model_*..safetensors"),
            key=lambda x: float(x.stem.split("_")[-1]),
        )
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

        for layer_name in layer_learnable_parameters:
            param = layer_learnable_parameters[layer_name]
            for pattern in self.config.patterns:
                if fnmatch(layer_name, pattern) and param.grad is not None:
                    norm = compute_grad_norm([param])
                    trainer.wandb_log(data={f"layer_grad_norm/{layer_name}": norm})
        
        if self.config.total:
            trainer.wandb_log(data={"grad_norm": trainer.total_gradient_norm})

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
            loss = trainer.loss
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) / self.approximate_loss(inverse_timestep)
            loss = loss.mean()
            trainer.loss = loss
        else:
            trainer.loss = trainer.loss.mean()

# class MmdEvaluation(Callback[Any]):
#     def eval_dataset(self, trainer: "PaletteTrainer") -> None:
#         pass
#     def on_evaluation_start(self, trainer: "PaletteTrainer") -> None:
#         trainer.eval_da
#         trainer.wandb_log(data={"mmd": trainer.mmd.item()})
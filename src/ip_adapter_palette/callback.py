from pathlib import Path
from typing import Any
from refiners.training_utils.common import (
    count_learnable_parameters,
    human_readable_number
)
from refiners.training_utils.callback import Callback, CallbackConfig
from loguru import logger

class OffloadToCPUConfig(CallbackConfig):
    use: bool = False

class OffloadToCPU(Callback[Any]):
    def on_init_end(self, trainer: Any) -> None:
        if trainer.config.offload_to_cpu.use:
            trainer.lda.to("cpu")
            trainer.text_encoder.to("cpu")
            trainer.ip_adapter.clip_image_encoder.to("cpu")

    def on_train_begin(self, trainer: Any) -> None:
        trainer.ip_adapter.inject()

class LogModelParamConfig(CallbackConfig):
    use: bool = True

class LogModelParam(Callback[Any]):
    def on_init_end(self, trainer: Any) -> None:
        for key in trainer.models.keys():
            logger.info(f"{key} : {human_readable_number(number=count_learnable_parameters(trainer.models[key].learnable_parameters))}")

class SaveBestModelConfig(CallbackConfig):
    max_checkpoints: int = 5


class SaveBestModel(Callback[Any]):
    current_epoch_losses: list[float] = []

    def __init__(self, config: SaveBestModelConfig) -> None:
        self.best_loss = float("inf")
        self.max_checkpoints = config.max_checkpoints

    def on_compute_loss_end(self, trainer: Any) -> None:
        self.current_epoch_losses.append(trainer.loss)

    def on_epoch_start(self, trainer: Any) -> None:
        self.current_epoch_losses = []

    def on_epoch_end(self, trainer: Any) -> None:
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

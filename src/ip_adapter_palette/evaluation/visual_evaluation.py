from functools import cached_property
from typing import Any, Literal
from loguru import logger
import numpy as np
from requests import get
from refiners.fluxion import load_from_safetensors
from refiners.fluxion.utils import no_grad
from sklearn.metrics import davies_bouldin_score
from ip_adapter_palette.metrics.mmd import mmd
from ip_adapter_palette.palette_adapter import Palette, Color
from ip_adapter_palette.histogram import histogram_to_histo_channels
from ip_adapter_palette.metrics.palette import batch_image_palette_metrics, ImageAndPalette
from ip_adapter_palette.evaluation.utils import BatchOutput, build_results, draw_histogram_cover_image, draw_palette_cover_image

from refiners.training_utils import (
    register_model,
    register_callback,
)
import os
from ip_adapter_palette.datasets import ColorDataset, GridEvalDataset, ColorIndexesDataset
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet

from refiners.training_utils.wandb import WandbLoggable
from torch import Tensor, tensor, randn, cat
from ip_adapter_palette.types import BatchInput
from refiners.training_utils.huggingface_datasets import load_hf_dataset, HuggingfaceDatasetConfig

from torch.utils.data import DataLoader

from datasets import load_dataset# type: ignore
from loguru import logger
from PIL import Image, ImageDraw
from refiners.training_utils.common import scoped_seed
from refiners.fluxion.utils import tensor_to_images, tensor_to_image, images_to_tensor
from torch.nn.functional import mse_loss
from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.foundationals.latent_diffusion import LatentDiffusionAutoencoder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ip_adapter_palette.trainer import PaletteTrainer

class VisualEvaluationConfig(CallbackConfig):
    condition_scale: float = 7.5
    batch_size: int = 1
    db_indexes: list[int] = []
    use: bool = False
    use_unconditional_text_embedding: bool = False

class VisualEvaluationCallback(Callback[Any]):
    def __init__(self, config: VisualEvaluationConfig) -> None:
        self.config = config
        self.batch_size = config.batch_size
        self.use = config.use
        self.db_indexes = config.db_indexes
        super().__init__()
    
    def on_init_end(self, trainer: Any) -> None:
        if not self.use:
            return
        self.dataset = ColorIndexesDataset(
            hf_dataset_config=trainer.config.eval_dataset,
            lda=trainer.lda,
            text_encoder=trainer.text_encoder,
            palette_extractor_weighted=trainer.palette_extractor_weighted,
            histogram_extractor=trainer.histogram_extractor,
            folder=trainer.config.data,
            db_indexes=self.db_indexes,
            pixel_sampler=trainer.pixel_sampler,
            spatial_tokenizer=trainer.spatial_tokenizer,
        )

        logger.info(f"Visual Evaluation activated with {len(self.db_indexes)} samples")
        
        self.dataloader = DataLoader(
            dataset=self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=BatchInput.collate, 
        )
    
    def on_precompute_start(self, trainer: "PaletteTrainer") -> None:
        self.dataset.precompute_embeddings()
    
    def on_evaluate_begin(self, trainer: "PaletteTrainer") -> None:
        self.compute_visual_evaluation(trainer)
    
    def compute_visual_evaluation(
        self,
        trainer: "PaletteTrainer"
    ) -> None:
        results_list : list[BatchOutput] = []

        for batch in self.dataloader:
            result_latents = trainer.batch_inference(
                batch.to(device=trainer.device, dtype=trainer.dtype),
                condition_scale=self.config.condition_scale,
                use_unconditional_text_embedding=self.config.use_unconditional_text_embedding
            )
            results_list.append(build_results(batch, result_latents, trainer, self.dataset))

        all_results = BatchOutput.collate(results_list)

        images : dict[str, WandbLoggable] = {}

        for result in all_results:
            image = draw_palette_cover_image(result, trainer)
            image_name = f"visual_eval/{result.source_prompts[0][0:100]}"
            images[image_name] = image

        trainer.wandb_log(data=images)
            

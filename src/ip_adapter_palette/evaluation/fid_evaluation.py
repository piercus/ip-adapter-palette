from functools import cached_property
from typing import Any, Literal
from unittest import result
from loguru import logger
import numpy as np
from requests import get
from refiners.fluxion import load_from_safetensors
from refiners.fluxion.utils import no_grad
from ip_adapter_palette.metrics.mmd import mmd
from ip_adapter_palette.palette_adapter import Palette, Color
from ip_adapter_palette.histogram import histogram_to_histo_channels
from ip_adapter_palette.metrics.palette import batch_image_palette_metrics, ImageAndPalette
from ip_adapter_palette.evaluation.utils import get_eval_images
from refiners.training_utils import (
    register_model,
    register_callback,
)
import os
from ip_adapter_palette.datasets import ColorDataset, GridEvalDataset
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
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
from refiners.fluxion.utils import image_to_tensor, normalize
from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ip_adapter_palette.trainer import PaletteTrainer

from torcheval.metrics import FrechetInceptionDistance

class FidEvaluationConfig(CallbackConfig):
    condition_scale: float = 7.5
    use_unconditional_text_embedding: bool = False
    batch_size: int = 1
    use: bool = False

class FidEvaluationCallback(Callback[Any]):
    def __init__(self, config: FidEvaluationConfig) -> None:
        self.config = config
        self.batch_size = config.batch_size
        self.use = config.use
        super().__init__()
    
    @cached_property
    def lda(self) -> LatentDiffusionAutoencoder:
        return self.trainer.lda
    
    def on_init_end(self, trainer: Any) -> None:
        if not self.use:
            return
        
        self.trainer = trainer
        self.cache_db_fid = FrechetInceptionDistance(device=self.device)
        self.fid = FrechetInceptionDistance(device=self.device)
        self.dtype = trainer.dtype
        self.device = trainer.device
        self.dataset = ColorDataset(
            hf_dataset_config=trainer.config.eval_dataset,
            lda=trainer.lda,
            text_encoder=trainer.text_encoder,
            palette_extractor_weighted=trainer.palette_extractor_weighted,
            histogram_extractor=trainer.histogram_extractor,
            folder=trainer.config.data,
            pixel_sampler=trainer.pixel_sampler,
            spatial_tokenizer=trainer.spatial_tokenizer,
        )

        logger.info(f"FID Evaluation activated with {len(self.dataset)} samples.")
        
        self.dataloader = DataLoader(
            dataset=self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=BatchInput.collate, 
        )
        logger.info(f"FID expected database precomputing")

        for batch in self.dataloader:
            source_images=get_eval_images(batch.db_indexes, batch.photo_ids, self.dataset)
            self.cache_db_fid.update(source_images, True)
        
        logger.info(f"FID expected database precomputing done")

    def on_precompute_start(self, trainer: "PaletteTrainer") -> None:
        if not self.use:
            return
        self.dataset.precompute_embeddings()
    
    def on_evaluate_begin(self, trainer: "PaletteTrainer") -> None:
        if not self.use:
            return
        logger.info("Starting FID evaluation")
        self.compute_fid_evaluation(trainer)
    
    def compute_fid_evaluation(
        self,
        trainer: "PaletteTrainer"
    ) -> None:
        self.fid.reset()

        for batch in self.dataloader:
            result_latents = trainer.batch_inference(
                batch.to(device=trainer.device, dtype=trainer.dtype),
                condition_scale=self.config.condition_scale,
                use_unconditional_text_embedding=self.config.use_unconditional_text_embedding
            )
            result_images = self.lda.latents_to_images(result_latents)
            self.fid.update(result_images, False)
        
        self.fid.merge_state(self.cache_db_fid)
        metric = self.fid.compute()
        trainer.wandb_log({
            "fid": metric
        })
        
        



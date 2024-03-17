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

from ip_adapter_palette.utils import preprocess_image
if TYPE_CHECKING:
    from ip_adapter_palette.trainer import PaletteTrainer

class BatchCLIPOutput(BatchInput):   
    result_clip: Tensor
    source_clip: Tensor

class MmdEvaluationConfig(CallbackConfig):
    condition_scale: float = 7.5
    batch_size: int = 1
    use_unconditional_text_embedding: bool = False
    use: bool = False

class MmdEvaluationCallback(Callback[Any]):
    def __init__(self, config: MmdEvaluationConfig) -> None:
        self.config = config
        self.batch_size = config.batch_size
        self.use = config.use
        super().__init__()
    
    @cached_property
    def clip_image_encoder(self) -> CLIPImageEncoderH:
        return CLIPImageEncoderH(
            dtype = self.dtype,
            device = self.device,
        )
    
    def encode_images(self, images: list[Image.Image]) -> Tensor:
        return self.clip_image_encoder(cat([preprocess_image(image, device=self.device, dtype=self.dtype) for image in images]))
    
    @cached_property
    def lda(self) -> LatentDiffusionAutoencoder:
        return self.trainer.lda
    
    def on_init_end(self, trainer: Any) -> None:
        if not self.use:
            return
        
        self.trainer = trainer

        self.dtype = trainer.dtype
        self.device = trainer.device
        self.dataset = ColorDataset(
            hf_dataset_config=trainer.config.eval_dataset,
            lda=trainer.lda,
            text_encoder=trainer.text_encoder,
            image_encoder=trainer.image_encoder,
            palette_extractor_weighted=trainer.palette_extractor_weighted,
            histogram_extractor=trainer.histogram_extractor,
            folder=trainer.config.data,
            pixel_sampler=trainer.pixel_sampler,
            spatial_tokenizer=trainer.spatial_tokenizer,
        )

        logger.info(f"MMD Evaluation activated with {len(self.dataset)} samples.")
        
        self.dataloader = DataLoader(
            dataset=self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=BatchInput.collate, 
        )
    
    def on_precompute_start(self, trainer: "PaletteTrainer") -> None:
        self.dataset.precompute_embeddings()
    
    def on_evaluate_begin(self, trainer: "PaletteTrainer") -> None:
        if not self.use:
            return        
        logger.info("Starting MMD evaluation")
        self.compute_mmd_evaluation(trainer)

    def build_results(self, batch: BatchInput, result_latents: Tensor) -> BatchCLIPOutput:
        source_images=get_eval_images(batch.db_indexes, batch.photo_ids, self.dataset)

        return BatchCLIPOutput(
            source_prompts=batch.source_prompts,
            source_histograms=batch.source_histograms,
            source_palettes_weighted=batch.source_palettes_weighted,
            source_text_embedding= batch.source_text_embedding,
            source_image_embedding=batch.source_image_embedding,
            source_bw_image_embedding=batch.source_bw_image_embedding,
            source_random_embedding=batch.source_random_embedding,
            source_random_long_embedding=batch.source_random_long_embedding,
            source_latents=batch.source_latents,
            db_indexes=batch.db_indexes,
            photo_ids=batch.photo_ids,
            source_clip=self.encode_images(source_images),
            result_clip=self.encode_images(self.lda.latents_to_images(result_latents)),
            source_pixel_sampling=batch.source_pixel_sampling,
            source_spatial_tokens=batch.source_spatial_tokens
        )
    
    def compute_mmd_evaluation(
        self,
        trainer: "PaletteTrainer"
    ) -> None:
        results_list : list[BatchCLIPOutput] = []

        for batch in self.dataloader:
            result_latents = trainer.batch_inference(
                batch.to(device=trainer.device, dtype=trainer.dtype),
                condition_scale=self.config.condition_scale,
                use_unconditional_text_embedding=self.config.use_unconditional_text_embedding
            )
            results_list.append(self.build_results(batch, result_latents))
        
        all_results = BatchCLIPOutput.collate(results_list)

        mmd_score = mmd(
            all_results.source_clip,
            all_results.result_clip
        ).item()

        trainer.wandb_log({
            "mmd": mmd_score
        })

        logger.info(f"MMD done : {mmd_score}")
        

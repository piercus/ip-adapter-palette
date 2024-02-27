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

class BatchCLIPOutput(BatchInput):   
    result_clip: Tensor
    source_clip: Tensor

class MmdEvaluationConfig(CallbackConfig):
    condition_scale: float = 7.5
    batch_size: int = 1
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
        return self.clip_image_encoder(cat([self.preprocess_image(image) for image in images]))
    
    def preprocess_image(
        self,
        image: Image.Image,
        size: tuple[int, int] = (224, 224),
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> Tensor:
        """Preprocess the image.

        Note:
            The default mean and std are parameters from
            https://github.com/openai/CLIP

        Args:
            image: The image to preprocess.
            size: The size to resize the image to.
            mean: The mean to use for normalization.
            std: The standard deviation to use for normalization.
        """
        return normalize(
            image_to_tensor(image.resize(size), dtype = self.dtype, device = self.device),
            mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
            std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
        )
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
            palette_extractor_weighted=trainer.palette_extractor_weighted,
            histogram_extractor=trainer.histogram_extractor,
            folder=trainer.config.data,
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
        logger.info("Starting MMD evaluation")
        self.compute_mmd_evaluation(trainer)

    def build_results(self, batch: BatchInput, result_latents: Tensor) -> BatchCLIPOutput:
        source_images=get_eval_images(batch.db_indexes, batch.photo_ids, self.dataset)

        return BatchCLIPOutput(
            source_prompts=batch.source_prompts,
            source_histograms=batch.source_histograms,
            source_palettes_weighted=batch.source_palettes_weighted,
            source_text_embeddings= batch.source_text_embeddings,
            source_latents=batch.source_latents,
            result_latents=result_latents,
            db_indexes=batch.db_indexes,
            photo_ids=batch.photo_ids,
            source_clip=self.encode_images(source_images),
            result_clip=self.encode_images(self.lda.latents_to_images(result_latents))
        )
    
    def compute_mmd_evaluation(
        self,
        trainer: "PaletteTrainer"
    ) -> None:
        results_list : list[BatchCLIPOutput] = []

        for batch in self.dataloader:
            result_latents = trainer.batch_inference(batch.to(device=trainer.device, dtype=trainer.dtype))
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
        

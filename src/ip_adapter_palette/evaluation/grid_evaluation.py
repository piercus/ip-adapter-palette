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

from refiners.training_utils import (
    register_model,
    register_callback,
)
import os
from ip_adapter_palette.datasets import EmbeddableDataset, GridEvalDataset
from ip_adapter_palette.evaluation.utils import BatchOutput, build_results, draw_histogram_cover_image, draw_palette_cover_image

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

class GridEvaluationConfig(CallbackConfig):
    use : bool = False
    db_indexes: list[int] = []
    prompts: list[str] = []
    condition_scale: float = 7.5
    batch_size: int = 1
    color_bits: int = 4
    use_unconditional_text_embedding: bool = False
    mode: Literal['palette', 'histogram'] = 'palette'


class GridEvaluationCallback(Callback[Any]):

    def __init__(self, config: GridEvaluationConfig) -> None:
        self.config = config
        self.use = self.config.use
        self.prompts = self.config.prompts
        self.db_indexes = self.config.db_indexes
        self.condition_scale = self.config.condition_scale
        self.mode = self.config.mode
        super().__init__()
    
    def on_init_end(self, trainer: Any) -> None:
        if not self.use:
            return
        self.dataset = GridEvalDataset(
            hf_dataset_config=trainer.config.eval_dataset,
            lda=trainer.lda,
            text_encoder=trainer.text_encoder,
            palette_extractor_weighted=trainer.palette_extractor_weighted,
            histogram_extractor=trainer.histogram_extractor,
            folder=trainer.config.data,
            db_indexes=self.db_indexes,
            prompts=self.prompts,
            pixel_sampler=trainer.pixel_sampler,
            spatial_tokenizer=trainer.spatial_tokenizer,
        )

        logger.info(f"Grid Evaluation activated with {len(self.db_indexes)} x {len(self.prompts)} samples.")
        
        self.dataloader = DataLoader(
            dataset=self.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            collate_fn=BatchInput.collate, 
            #num_workers=num_workers
        )
    def on_precompute_start(self, trainer: "PaletteTrainer") -> None:
        if not self.use:
            return        
        self.dataset.precompute_embeddings()
    
    def on_evaluate_begin(self, trainer: "PaletteTrainer") -> None:
        if not self.use:
            return        
        self.compute_grid_evaluation(trainer)

    def image_distances(self, batch: BatchOutput) -> float:
        images = images_to_tensor(batch.result_images)
        dist = tensor(0)
        for i in range(images.shape[0]):
            for j in range(i+1, images.shape[0]):
                dist = dist + mse_loss(images[i], images[j])
        
        return dist.item()

    def compute_grid_evaluation(
        self,
        trainer: "PaletteTrainer"
    ) -> None:
        
        per_prompts : dict[str, BatchOutput] = {}
        images : dict[str, WandbLoggable] = {}
                
        for batch in self.dataloader:
            result_latents = trainer.batch_inference(
                batch.to(device=trainer.device, dtype=trainer.dtype),
                condition_scale=self.config.condition_scale,
                use_unconditional_text_embedding=self.config.use_unconditional_text_embedding
            )
            results = build_results(batch, result_latents, trainer, self.dataset)
        
            for prompt in list(set(results.source_prompts)):
                batch = results.get_prompt(prompt)
                if prompt not in per_prompts:
                    per_prompts[prompt] = batch
                else:
                    per_prompts[prompt] = BatchOutput.collate([
                        per_prompts[prompt],
                        batch
                    ])
        
        for prompt in per_prompts:
            trainer.wandb_log(data={
                f"inter_prompt_distance/{prompt}": self.image_distances(per_prompts[prompt])
            })
            if self.mode == "histogram":
                image = draw_histogram_cover_image(per_prompts[prompt], trainer)
            else:
                image = draw_palette_cover_image(per_prompts[prompt], trainer)
            image_name = f"eval_images/{prompt[0:100]}"
            images[image_name] = image
            
        all_results = BatchOutput.collate(list(per_prompts.values()))
        
        # images[f"eval_images/all"] = self.draw_cover_image(all_results)
        trainer.wandb_log(data=images)

        self.batch_metrics(all_results, trainer, prefix="eval")
    
    def batch_metrics(self, results: BatchOutput, trainer: 'PaletteTrainer', prefix: str = "palette-img") -> None:
        palettes : list[list[Color]] = []
        for p in results.source_palettes_weighted:
            palettes.append([cluster[0] for cluster in p])
        
        images = results.result_images
        
        batch_image_palette_metrics(
            trainer.wandb_log, 
            [
                ImageAndPalette({"image": image, "palette": palette})
                for image, palette in zip(images, palettes)
            ], 
            prefix
        )

        result_histo = results.result_histograms
        
        histo_metrics = trainer.histogram_distance.metrics(result_histo, results.source_histograms.to(result_histo.device))
        
        log_dict : dict[str, WandbLoggable] = {}
        for (key, value) in histo_metrics.items():
            log_dict[f"eval_histo/{key}"] = value.item()
        
        trainer.wandb_log(log_dict)  
    


    
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
from ip_adapter_palette.datasets import GridEvalDataset
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

from ip_adapter_palette.datasets import EmbeddableDataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ip_adapter_palette.trainer import PaletteTrainer

class BatchOutput(BatchInput):   
    source_images: list[Image.Image]
    result_latents: Tensor
    result_images: list[Image.Image]
    result_palettes_weighted: list[Palette]
    result_histograms: Tensor

def get_eval_images(db_indexes: list[int], photo_ids: list[str], dataset: EmbeddableDataset) -> list[Image.Image]:
    images = []
    for (db_index, photo_id) in zip(db_indexes, photo_ids):
        hf_item = dataset.get_processed_hf_item(db_index)
        if hf_item['photo_id'] != photo_id:
            raise ValueError(f"Photo id mismatch: {hf_item['photo_id']} != {photo_id}")
        images.append(hf_item['image'])
    return images

def build_results(batch: BatchInput, result_latents: Tensor, trainer: 'PaletteTrainer', dataset: EmbeddableDataset) -> BatchOutput:
    result_images = trainer.lda.latents_to_images(result_latents)
    return BatchOutput(
        source_prompts=batch.source_prompts,
        db_indexes=batch.db_indexes,
        source_histograms=batch.source_histograms,
        source_palettes_weighted=batch.source_palettes_weighted,
        source_images=get_eval_images(batch.db_indexes, batch.photo_ids, dataset),
        photo_ids = batch.photo_ids,
        source_text_embeddings= batch.source_text_embeddings,
        source_latents=batch.source_latents,
        result_histograms = trainer.histogram_extractor.images_to_histograms(result_images),
        result_latents = result_latents,
        result_images=result_images,         
        result_palettes_weighted=[trainer.palette_extractor_weighted(image, size=len(batch.source_palettes_weighted[i])) for i, image in enumerate(result_images)],
    )

def draw_palette(palette: Palette, width: int, height: int) -> Image.Image:
    palette_img = Image.new(mode="RGB", size=(width, height))
    
    # sort the palette by weight
    current_x = 0
    for (color, weight) in palette:
        box_width = int(weight*width)            
        color_box = Image.fromarray(np.full((height, box_width, 3), color, dtype=np.uint8)) # type: ignore
        palette_img.paste(color_box, box=(current_x, 0))
        current_x+=box_width
        
    return palette_img

def draw_palette_cover_image(batch: BatchOutput, trainer: 'PaletteTrainer') -> Image.Image:
    res_images = images_to_tensor(batch.result_images)
    (batch_size, _, height, width) = res_images.shape
    
    palette_img_size = width // trainer.config.palette_encoder.max_colors
    source_images = batch.source_images

    join_canvas_image: Image.Image = Image.new(
        mode="RGB", size=(2*width, (height+palette_img_size) * batch_size)
    )
    
    images = tensor_to_images(res_images)
    sources_palettes_processed = trainer.process_palettes(batch.source_palettes_weighted)
    result_palettes_processed = trainer.process_palettes(batch.result_palettes_weighted)
    for i, image in enumerate(images):
        join_canvas_image.paste(source_images[i], box=(0, i*(height+palette_img_size)))
        join_canvas_image.paste(image, box=(width, i*(height+palette_img_size)))
        palette_out = result_palettes_processed[i]
        palette_int = result_palettes_processed[i]
        palette_out_img = draw_palette(palette_out, width, palette_img_size)
        palette_in_img = draw_palette(palette_int, width, palette_img_size)
        
        join_canvas_image.paste(palette_in_img, box=(0, i*(height+palette_img_size) + height))
        join_canvas_image.paste(palette_out_img, box=(width, i*(height+palette_img_size) + height))
    return join_canvas_image


def draw_curves(res_histo: list[float], src_histo: list[float], color: str, width: int, height: int) -> Image.Image:
    histo_img = Image.new(mode="RGB", size=(width, height))
    
    draw = ImageDraw.Draw(histo_img)
    
    if len(res_histo) != len(src_histo):
        raise ValueError("The histograms must have the same length.")
    
    ratio = width/len(res_histo)
    semi_height = height//2
    
    scale_ratio = 5
            
    draw.line([
        (i*ratio, (1-res_histo[i]*scale_ratio)*semi_height + semi_height) for i in range(len(res_histo))
    ], fill=color, width=4)
    
    draw.line([
        (i*ratio, (1-src_histo[i]*scale_ratio)*semi_height) for i in range(len(src_histo))
    ], fill=color, width=1)
    
    return histo_img

def draw_histogram_cover_image(batch: BatchOutput, trainer: 'PaletteTrainer') -> Image.Image:
    res_images = images_to_tensor(batch.result_images)
    (batch_size, channels, height, width) = res_images.shape

    vertical_image = res_images.permute(0,2,3,1).reshape(1, height*batch_size, width, channels).permute(0,3,1,2)
    
    results_histograms = batch.result_histograms
    source_histograms = batch.source_histograms
    source_images = batch.source_images
    src_palettes = batch.source_palettes_weighted

    join_canvas_image: Image.Image = Image.new(
        mode="RGB", size=(width + width//2, height * batch_size)
    )
    res_image = tensor_to_image(vertical_image)
    
    join_canvas_image.paste(res_image, box=(width//2, 0))
    
    res_histo_channels = histogram_to_histo_channels(results_histograms)
    src_histo_channels = histogram_to_histo_channels(source_histograms)
    
    colors = ["red", "green", "blue"]
    
    for i in range(batch_size):
        image = source_images[i]
        join_canvas_image.paste(image.resize((width//2, height//2)), box=(0, height *i))
        
        source_image_palette = draw_palette(
            trainer.palette_extractor_weighted.from_histogram(source_histograms[i], color_bits= trainer.config.histogram_auto_encoder.color_bits, size=len(src_palettes[i])),
            width//2,
            height//16
        )
        join_canvas_image.paste(source_image_palette, box=(0, height *i + height//2))
        
        res_image_palette = draw_palette(
            trainer.palette_extractor_weighted.from_histogram(results_histograms[i], color_bits= trainer.config.histogram_auto_encoder.color_bits, size=len(src_palettes[i])),
            width//2,
            height//16
        )
        
        join_canvas_image.paste(res_image_palette, box=(0, height *i + (15*height)//16))

        for (color_id, color_name) in enumerate(colors):
            image_curve = draw_curves(
                res_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                src_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                color_name,
                width//2,
                height//8
            )
            join_canvas_image.paste(image_curve, box=(0, height *i + height//2 + ((1+2*color_id)*height)//16))
            
    return join_canvas_image
from calendar import c
from ip_adapter_palette.utils import AbstractBatchInput, AbstractBatchOutput, Batch
from PIL import Image
from typing import TypedDict
from torch import load as torch_load, Tensor
Color = tuple[int, int, int]
ColorWeight = float
PaletteCluster = tuple[Color, ColorWeight]
Palette = list[PaletteCluster]
from pathlib import Path

class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: list[Color]

class BatchInput(Batch):
    source_palettes: list[Palette]
    source_prompts: list[str]
    db_indexes: list[int]
    photo_ids: list[str]
    source_images: Tensor
    source_text_embeddings: Tensor
    source_latents: Tensor
    source_histograms: Tensor

    @classmethod
    def load_file(cls, filename: Path) -> "BatchInput":
        return cls(**torch_load(filename))
    
    def id(self) -> str:
        return '_'.join(self.photo_ids)

class BatchOutput(Batch):    
    source_palettes: list[Palette]
    source_prompts: list[str]
    source_images: Tensor
    source_histograms: Tensor
    db_indexes: list[int]
    source_latents: Tensor
    text_embeddings: Tensor
    latents: Tensor
    result_images: Tensor
    result_palettes: list[Palette]
    result_histograms: list[Tensor]


from calendar import c
from ip_adapter_palette.utils import Batch
from PIL import Image
from typing import TypedDict, TypeVar
from torch import load as torch_load, Tensor
Color = tuple[int, int, int]
ColorWeight = float
PaletteCluster = tuple[Color, ColorWeight]
Palette = list[PaletteCluster]
from pathlib import Path

T = TypeVar('T', bound='BatchInput')

class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: list[Color]

class BatchInput(Batch):
    source_palettes_weighted: list[Palette]
    source_prompts: list[str]
    db_indexes: list[int]
    photo_ids: list[str]
    source_text_embeddings: Tensor
    source_latents: Tensor
    source_histograms: Tensor

    @classmethod
    def load_file(cls, filename: Path) -> "BatchInput":
        return cls(**torch_load(filename, map_location='cpu'))
    
    def id(self) -> str:
        return '_'.join(self.photo_ids)
    
    def get_prompt(self: T, prompt: str) -> "T":
        res : list[T] = []
        for single_batch in self:
            if prompt in single_batch.source_prompts:
                res.append(single_batch)
        return self.__class__.collate(res)





from refiners.training_utils.batch import BaseBatch # type: ignore
from PIL import Image
from typing import TypedDict, TypeVar
from torch import load as torch_load, Tensor
Color = tuple[int, int, int]
ColorWeight = float
PaletteCluster = tuple[Color, ColorWeight]
Palette = list[PaletteCluster]
Sample = list[Color]
from pathlib import Path

T = TypeVar('T', bound='BatchInput')

class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: list[Color]

class BatchInput(BaseBatch):
    source_palettes_weighted: list[Palette]
    source_prompts: list[str]
    db_indexes: list[int]
    photo_ids: list[str]
    source_text_embeddings: Tensor
    source_latents: Tensor
    source_histograms: Tensor
    source_pixel_sampling: Tensor
    source_spatial_tokens: Tensor

    def id(self) -> str:
        return '_'.join(self.photo_ids)
    # @classmethod
    # def load(cls, path: Path) -> "BatchInput":
    #     loaded = super().load(path)
        
    #     # hardcode fix
    #     loaded.source_pixel_sampling=loaded.source_pixel_sampling[:,0:2048,:]

    #     return loaded
    
    def get_prompt(self: T, prompt: str) -> "T":
        indices : list[int] = [
            index for index, p in enumerate(self.source_prompts) if p == prompt
        ]
        return self[indices]





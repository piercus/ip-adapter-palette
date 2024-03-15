from refiners.training_utils.batch import BaseBatch # type: ignore
from PIL import Image
from typing import TypedDict, TypeVar
from torch import load as torch_load, Tensor
from torch.nn import functional as F
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
    source_text_embedding: Tensor
    source_latents: Tensor
    source_histograms: Tensor
    source_pixel_sampling: Tensor
    source_spatial_tokens: Tensor
    source_random_embedding: Tensor
    source_random_long_embedding: Tensor
    source_image_embedding: Tensor
    source_bw_image_embedding: Tensor

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

class BatchInputProcessed(BatchInput):
    processed_text_embedding: Tensor
    processed_prompts: list[str]
    processed_image_embedding: Tensor
    processed_bw_image_embedding: Tensor
    @classmethod
    def from_batch_input_unconditionnal(cls: type[T], item: BatchInput, uncond_txt: Tensor, uncond_img: Tensor) -> "T":
        return cls(
            source_palettes_weighted=item.source_palettes_weighted,
            source_prompts=item.source_prompts,
            db_indexes=item.db_indexes,
            photo_ids=item.photo_ids,
            source_text_embedding=item.source_text_embedding,
            source_latents=item.source_latents,
            source_histograms=item.source_histograms,
            source_pixel_sampling=item.source_pixel_sampling,
            source_spatial_tokens=item.source_spatial_tokens,
            source_random_embedding=item.source_random_embedding,
            source_random_long_embedding=item.source_random_long_embedding,
            source_image_embedding=item.source_image_embedding,
            source_bw_image_embedding=item.source_bw_image_embedding,
            processed_image_embedding=uncond_img.repeat(len(item), 1, 1),
            processed_bw_image_embedding=uncond_img.repeat(len(item), 1, 1),
            processed_text_embedding=uncond_txt.repeat(len(item), 1, 1),
            processed_prompts=[""]*len(item)
        )
    @classmethod
    def from_batch_input(cls: type[T], item: BatchInput) -> "T":
        return cls(
            source_palettes_weighted=item.source_palettes_weighted,
            source_prompts=item.source_prompts,
            db_indexes=item.db_indexes,
            photo_ids=item.photo_ids,
            source_text_embedding=item.source_text_embedding,
            source_latents=item.source_latents,
            source_histograms=item.source_histograms,
            source_pixel_sampling=item.source_pixel_sampling,
            source_spatial_tokens=item.source_spatial_tokens,
            source_random_embedding=item.source_random_embedding,
            source_random_long_embedding=item.source_random_long_embedding,
            source_image_embedding=item.source_image_embedding,
            source_bw_image_embedding=item.source_bw_image_embedding,
            processed_text_embedding=item.source_text_embedding,
            processed_prompts=item.source_prompts,
            processed_image_embedding=item.source_image_embedding,
            processed_bw_image_embedding=item.source_bw_image_embedding
        )
    
    def test_processed_vs_source(self):
        return F.mse_loss(input=self.source_text_embedding, target=self.processed_text_embedding, reduction='mean').item()




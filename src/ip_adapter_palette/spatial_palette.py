
from PIL import Image
from typing import Literal
from functools import cached_property
from ip_adapter_palette.types import Sample
import numpy as np
import refiners.fluxion.layers as fl
from torch import cat, Tensor, tensor, ones, device as Device, dtype as DType, zeros
from jaxtyping import Float
from torch.nn.functional import pad
from ip_adapter_palette.palette_adapter import PaletteTransformerEncoder, PaletteMLPEncoder, ColorEncoder
from torchvision.transforms.functional import resize
from refiners.fluxion.utils import images_to_tensor
from refiners.foundationals.clip.common import PositionalEncoder
from refiners.foundationals.clip.image_encoder import TransformerLayer

class SpatialTokenizer(fl.Chain):
    def __init__(
        self,
        thumb_size: int = 8,
        input_size:int = 512,
        device: Device | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.thumb_size = thumb_size
        self.input_size = input_size
        self._device = device
        self._dtype = dtype
        super().__init__()
    
    def forward(self, images: list[Image.Image]) -> Float[Tensor, "*batch 3 thumb_size thumb_size"]:
        images = [image.convert("RGB") if image.mode != "RGB" else image for image in images]
        img_tensor = images_to_tensor(images, device=self._device, dtype=self._dtype)

        if img_tensor.shape[2] != img_tensor.shape[3]:
            raise ValueError("Images must be square")
        if img_tensor.shape[2] != self.input_size:
            raise ValueError(f"Images must be {self.input_size}x{self.input_size}")
        return resize(img_tensor, [self.thumb_size, self.thumb_size])

class ClassToken(fl.Chain):
    def __init__(self, embedding_dim: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.embedding_dim = embedding_dim
        super().__init__(fl.Parameter(1, embedding_dim, device=device, dtype=dtype))

class SpatialPaletteEncoder(fl.Chain):
    _sampler: list[SpatialTokenizer]

    def __init__(
        self,
        embedding_dim: int = 768,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        feedforward_dim: int = 20,
        layer_norm_eps: float = 1e-5,
        mode: str = 'transformer',
        tokenizer: SpatialTokenizer = SpatialTokenizer(),
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self._tokenizer = [tokenizer]

        self.embedding_dim = embedding_dim
        if num_layers == 0:
            encoder_body = fl.Identity()
        elif mode == 'transformer':
            encoder_body = fl.Chain(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    num_attention_heads=num_attention_heads,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            )
        else:
            raise ValueError(f"Unknown mode {mode}")
        
        self.thumb_size = self.tokenizer.thumb_size

        # Inspired from ViTEmbeddings
        super().__init__(
            fl.Concatenate(
                ClassToken(embedding_dim, device=device, dtype=dtype),
                fl.Chain(
                    fl.Reshape(self.thumb_size ** 2, 3),
                    ColorEncoder(
                        embedding_dim=embedding_dim,
                        device=device,
                        in_features=3,
                        weighted_palette=False,
                        use_lda=False,
                        dtype=dtype,
                    )
                ),
                dim=1,
            ),
            fl.Residual(
                PositionalEncoder(
                    max_sequence_length=self.thumb_size ** 2 + 1,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
            encoder_body,
        )
    
    def images_to_latents(self, images: list[Image.Image], sizes: list[int] | None = None) -> Tensor:
        sample = self.tokenizer(images)
        return self(sample.to(device=self.device, dtype=self.dtype))
    
    @property
    def tokenizer(self) -> SpatialTokenizer:
        return self._tokenizer[0]
    
    def unconditionnal(self) -> Tensor:
        return self(self.unconditional_tokens)

    @cached_property
    def unconditional_tokens(self) -> Tensor:
        return zeros(1, 3, self.thumb_size, self.thumb_size, device=self.device, dtype=self.dtype)
    
    def compute_spatial_palette_embedding(self, images: list[Image.Image], sizes: list[int] | None = None) -> Tensor:
        conditional_embedding =  self.images_to_latents(images)
        negative_embedding = self.unconditionnal().repeat(len(images), 1, 1, 1)
        return cat(tensors=(negative_embedding, conditional_embedding), dim=0)
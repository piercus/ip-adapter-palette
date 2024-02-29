
from PIL import Image
from typing import Literal
from ip_adapter_palette.types import Sample
import numpy as np
import refiners.fluxion.layers as fl
SamplerMode = Literal['first', 'random']
from torch import cat, Tensor, tensor, ones, device as Device, dtype as DType, zeros
from jaxtyping import Float
from torch.nn.functional import pad
from ip_adapter_palette.palette_adapter import PaletteTransformerEncoder, PaletteMLPEncoder, ColorEncoder

class Sampler(fl.Module):
    def __init__(
        self,
        max_size: int = 2048,
        mode: SamplerMode = 'first'
    ) -> None:
        self.max_size = max_size
        self.mode = mode
        super().__init__()
    
    def forward(self, images: list[Image.Image], sizes : list[int] | None = None) -> Float[Tensor, "*batch max_size 5"]:
        if sizes is None:
            sizes = [self.max_size] * len(images)
        
        if len(images) != len(sizes):
            raise ValueError("Images and sizes must have the same length")
        
        tensors = [self._forward(image, size) for image, size in zip(images, sizes)]
        return cat(tensors, dim=0)
    
    def _forward(self, image: Image.Image, size: int) -> Float[Tensor, "1 max_size 5"]:
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if size == 0:
            sample = zeros(1, 0, 3)
        elif self.mode == 'first':
            frequency = (image.size[0] * image.size[1]) // (size-1)
            pixels = list(image.getdata())[0::frequency]
            if len(pixels) > size:
                raise ValueError(f"Sample {pixels} is bigger than expected size {size}")
            sample = tensor([pixels])
        elif self.mode == 'random':
            image_np = np.array(image).reshape(-1, 3)
            pixels = image_np[np.random.choice(len(image_np), size, replace=False)].tolist()
            sample = tensor([pixels])

        sample = self.add_channel(sample)
        sample = self.zero_right_padding(sample)
        return sample

    def add_channel(self, x: Float[Tensor, "*batch colors 3"]) -> Float[Tensor, "*batch colors_with_end 4"]:
        return cat((x, ones(x.shape[0], x.shape[1], 1)), dim=2)

    def zero_right_padding(
        self, x: Float[Tensor, "*batch colors_with_end embedding_dim"]
    ) -> Float[Tensor, "*batch max_colors feedforward_dim"]:
        # Zero padding for the right side
        padding_width = (self.max_size - x.shape[1] % self.max_size) % self.max_size
        if x.shape[1] == 0:
            padding_width = self.max_size
        result = pad(x, (0, 0, 0, padding_width))
        return result


class Encoder(fl.Chain):
    _sampler: list[Sampler]

    def __init__(
        self,
        embedding_dim: int = 768,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        feedforward_dim: int = 20,
        layer_norm_eps: float = 1e-5,
        mode: str = 'transformer',
        sampler: Sampler = Sampler(),
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self._sampler = [sampler]
        self.embedding_dim = embedding_dim
        if num_layers == 0:
            encoder_body = fl.Identity()
        elif mode == 'transformer':
            encoder_body = PaletteTransformerEncoder(
                embedding_dim=embedding_dim,
                max_colors=self.sampler.max_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                feedforward_dim=feedforward_dim,
                layer_norm_eps=layer_norm_eps,
                device=device,
                dtype=dtype
            )
        elif mode == 'mlp':
            encoder_body = PaletteMLPEncoder(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                feedforward_dim=feedforward_dim,
                layer_norm_eps=layer_norm_eps,
                device=device,
                dtype=dtype
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

        super().__init__(
            ColorEncoder(
                embedding_dim=embedding_dim,
                device=device,
                weighted_palette=False,
                use_lda=False,
                dtype=dtype,
            ),
            encoder_body,
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
        )
    
    def images_to_latents(self, images: list[Image.Image], sizes: list[int] | None = None) -> Tensor:
        sample = self.sampler(images, sizes)
        return self(sample.to(device=self.device, dtype=self.dtype))
    
    @property
    def sampler(self) -> Sampler:
        return self._sampler[0]
    
    def compute_sampling_embedding(self, images: list[Image.Image], sizes: list[int] | None = None) -> Tensor:
        conditional_embedding =  self.images_to_latents(images, sizes)
        negative_embedding = self.images_to_latents(images, sizes = len(images) * [0])
        return cat(tensors=(negative_embedding, conditional_embedding), dim=0)
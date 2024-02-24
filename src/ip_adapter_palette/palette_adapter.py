from typing import Any, List, TypeVar, cast

from jaxtyping import Float
from torch import Tensor, cat, device as Device, dtype as DType, float32, ones, tensor, zeros
from torch.nn.functional import pad
from torch.nn import init

import refiners.fluxion.layers as fl
from ip_adapter_palette.types import Color, Palette, PaletteCluster
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.fluxion.layers.basics import Parameter
from refiners.foundationals.clip.common import FeedForward, PositionalEncoder
from refiners.foundationals.clip.text_encoder import TransformerLayer
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

TSDNet = TypeVar("TSDNet", bound="SD1UNet | SDXLUNet")

class PalettesTokenizer(fl.Module):
    _lda: list[SD1Autoencoder]

    @property
    def lda(self):
        return self._lda[0]

    def __init__(
        self,
        max_colors: int,
        lda: SD1Autoencoder,
        weighted_palette: bool = False,
        use_lda: bool = False
    ) -> None:
        self._lda = [lda]
        self.use_lda = use_lda
        self.weighted_palette = weighted_palette
        super().__init__()
        self.max_colors = max_colors
    
    def forward(self, palettes: List[Palette]) -> Float[Tensor, "*batch max_colors 5"]:
        tensors = [self._forward(palette) for palette in palettes]
        return cat(tensors, dim=0)

    def _forward(self, palette: Palette) -> Float[Tensor, "*batch max_colors 5"]:
        
        if len(palette) == 0:
            tensor_source = zeros(1, 0, 4)
        else:
            tensor_source = tensor([[[cluster[0][0], cluster[0][1], cluster[0][2], cluster[1]] for cluster in palette]], dtype=float32)
        
        tensor_weight = tensor_source[:, :, 3:4]
        tensor_colors = tensor_source[:, :, :3]

        if self.use_lda:
            tensor_colors = self.lda_encode(tensor_colors)
        
        if self.weighted_palette:
            tensor_palette = cat((tensor_colors, tensor_weight), dim=2)
        else:
            tensor_palette = tensor_colors
        
        colors = self.add_channel(tensor_palette)
        colors = self.zero_right_padding(colors)
        return colors

    def add_channel(self, x: Float[Tensor, "*batch colors 5"]) -> Float[Tensor, "*batch colors_with_end 6"]:
        return cat((x, ones(x.shape[0], x.shape[1], 1)), dim=2)

    def zero_right_padding(
        self, x: Float[Tensor, "*batch colors_with_end embedding_dim"]
    ) -> Float[Tensor, "*batch max_colors feedforward_dim"]:
        # Zero padding for the right side
        padding_width = (self.max_colors - x.shape[1] % self.max_colors) % self.max_colors
        if x.shape[1] == 0:
            padding_width = self.max_colors
        result = pad(x, (0, 0, 0, padding_width))
        return result

    def lda_encode(self, x: Float[Tensor, "*batch num_colors 3"]) -> Float[Tensor, "*batch num_colors 4"]:
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        num_colors = x.shape[1]
        if num_colors == 0:
            return x.reshape(batch_size, 0, 4)

        x = x.reshape(batch_size * num_colors, 3, 1, 1)
        x = x.repeat(1, 1, 8, 8).to(self.lda.device, self.lda.dtype)

        out = self.lda.encode(x).to(device, dtype)

        out = out.reshape(batch_size, num_colors, 4)
        return out

class ColorEncoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        weighted_palette: bool = False,
        use_lda : bool = False,
        device: Device | str | None = None,
        eps: float = 1e-5,
        dtype: DType | None = None,
    ) -> None:
        in_features = 4
        if use_lda:
            in_features += 1
        if weighted_palette:
            in_features += 1
        super().__init__(
            fl.Linear(in_features=in_features, out_features=embedding_dim, bias=True, device=device, dtype=dtype),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=eps, device=device, dtype=dtype),
        )

class PaletteTransformerEncoder(fl.Chain):

    def __init__(
        self,
        embedding_dim: int = 768,
        max_colors: int = 8,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        feedforward_dim: int = 20,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        # self._lda = [lda]
        self.embedding_dim = embedding_dim
        self.max_colors = max_colors
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        super().__init__(
            *(
                # Remark :
                # The current transformer layer has a causal self-attention
                # It would be fair to test non-causal self-attention
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    num_attention_heads=num_attention_heads,
                    feedforward_dim=feedforward_dim,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            )
        )

class PaletteMLPEncoder(fl.Chain):

    def __init__(
        self,
        embedding_dim: int = 768,
        num_layers: int = 2,
        feedforward_dim: int = 20,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        # self._lda = [lda]
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        
        super().__init__(
            *(
                fl.Chain(
                    fl.LayerNorm(
                        normalized_shape=embedding_dim,
                        eps=layer_norm_eps,
                        device=device,
                        dtype=dtype,
                    ),
                    FeedForward(
                        embedding_dim=embedding_dim,
                        feedforward_dim=feedforward_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                for _ in range(num_layers)
            )
        )




import numpy as np
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore


class PaletteExtractor:
    def __init__(
        self,
        size: int = 8,
        weighted_palette: bool = False,
    ) -> None:
        self.size = size
        self.weighted_palette = weighted_palette

    def __call__(self, image: Image.Image, size: int | None = None) -> Palette:
        if size is None:
            size = self.size
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image_np = np.array(image)
        pixels = image_np.reshape(-1, 3)
        return self.from_pixels(pixels, size)

    def from_pixels(self, pixels: np.ndarray[int, Any], size: int | None = None, eps : float = 1e-7) -> Palette:
        kmeans = KMeans(n_clusters=size).fit(pixels) # type: ignore 
        counts = np.unique(kmeans.labels_, return_counts=True)[1] # type: ignore
        palette : Palette = []
        if size is None:
            size = self.size
        total = pixels.shape[0]
        for i in range(0, len(counts)):
            center_float : tuple[float, float, float] = kmeans.cluster_centers_[i] # type: ignore
            center : Color = tuple(center_float.astype(int)) # type: ignore
            count = float(counts[i].item())
            color_cluster: PaletteCluster = (
                center,
                count / total if self.weighted_palette else 1.0 / size
            )
            palette.append(color_cluster)
        
        if len(counts) < size:
            for _ in range(size - len(counts)):
                pal : PaletteCluster = ((0, 0, 0), eps if self.weighted_palette else 1.0 / size)
                palette.append(pal)
        sorted_palette = sorted(palette, key=lambda x: x[1], reverse=True)
        return sorted_palette

    def from_histogram(self, histogram: Tensor, color_bits: int, size: int | None = None, num: int = 1) -> Palette:
        if histogram.dim() != 4:
            raise Exception('histogram must be 4 dimensions')
        cube_size = 2 ** color_bits
        color_factor = 256 / cube_size
        pixels : list[np.ndarray[int, Any]] = []
        for histo in histogram.split(1): # type: ignore
            for r in range(cube_size):
                for g in range(cube_size):
                    for b in range(cube_size):
                        for _ in range(int(histo[0, r, g, b]* num)): # type: ignore
                            pixels.append(np.array([r*color_factor, g*color_factor, b*color_factor]))                
            
        return self.from_pixels(np.array(pixels), size)
    
    def distance(self, a: Palette, b: Palette) -> float:
        #TO DO
        raise NotImplementedError

class PaletteEncoder(fl.Chain):
    def __init__(
        self,
        lda: SD1Autoencoder = SD1Autoencoder(),
        embedding_dim: int = 768,
        max_colors: int = 8,
        # Remark :
        # I have followed the CLIPTextEncoderL parameters
        # as default parameters here, might require some testing
        num_layers: int = 2,
        num_attention_heads: int = 2,
        feedforward_dim: int = 20,
        layer_norm_eps: float = 1e-5,
        mode: str = 'transformer',
        use_lda: bool = False,
        weighted_palette: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        if num_layers == 0:
            encoder_body = fl.Identity()
        elif mode == 'transformer':
            encoder_body = PaletteTransformerEncoder(
                embedding_dim=embedding_dim,
                max_colors=max_colors,
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
        
        if weighted_palette:
            encoder = ColorEncoder(
                embedding_dim=embedding_dim,
                device=device,
                weighted_palette=weighted_palette,
                use_lda=use_lda,
                dtype=dtype,
            )
        else:
            encoder =  fl.Sum(
                ColorEncoder(
                    embedding_dim=embedding_dim,
                    device=device,
                    use_lda=use_lda,
                    weighted_palette=weighted_palette,
                    dtype=dtype,
                ),
                PositionalEncoder(
                    max_sequence_length=max_colors,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            )

        super().__init__(
            PalettesTokenizer(
                max_colors=max_colors,
                weighted_palette=weighted_palette,
                lda=lda,
                use_lda=use_lda,
            ),
            fl.Converter(),
            encoder,
            encoder_body,
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
        )
    
    def compute_palette_embedding(
        self,
        x: List[Palette] = [],
        negative_palette: List[Palette] | None = None,
    ) -> Float[Tensor, "cfg_batch n_colors 3"]:
        
        conditional_embedding = self(x)
        if negative_palette is None:
            negative_palette = [[]]*len(x)
        
        if len(negative_palette) != len(x):
            raise ValueError("The negative_palette must have the same length as the input color palette")
        
        negative_embedding = self(negative_palette)
        return cat(tensors=(negative_embedding, conditional_embedding), dim=0)
    
class PaletteCrossAttention(fl.Chain):
    def __init__(self, text_cross_attention: fl.Attention, embedding_dim: int = 768, scale: float = 1.0) -> None:
        self._scale = scale
        super().__init__(
            fl.Distribute(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="palette_embedding"),
                    fl.Linear(
                        in_features=embedding_dim,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="palette_embedding"),
                    fl.Linear(
                        in_features=embedding_dim,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
            ),
            ScaledDotProductAttention(
                num_heads=text_cross_attention.num_heads, is_causal=text_cross_attention.is_causal
            ),
            fl.Multiply(self.scale),
        )

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(fl.Multiply).scale = value
    
    def weights(self) -> list[Parameter]:
        return cast(list[Parameter], list(self.ensure_find(fl.Linear).parameters()))


class PaletteCrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(self, target: fl.Attention, scale: float = 1.0, embedding_dim: int = 768) -> None:
        self._scale = scale
        with self.setup_adapter(target):
            clone = target.structural_copy()
            scaled_dot_product = clone.ensure_find(ScaledDotProductAttention)
            palette_cross_attention = PaletteCrossAttention(
                text_cross_attention=clone,
                embedding_dim=embedding_dim,
                scale=self.scale,
            )
            clone.replace(
                old_module=scaled_dot_product,
                new_module=fl.Sum(
                    scaled_dot_product,
                    palette_cross_attention,
                ),
            )
            super().__init__(
                clone,
            )

    @property
    def palette_cross_attention(self) -> PaletteCrossAttention:
        return self.ensure_find(PaletteCrossAttention)
    
    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.palette_cross_attention.scale = value
    
    @property
    def image_key_projection(self) -> fl.Linear:
        return self.palette_cross_attention.layer(("Distribute", 1, "Linear"), fl.Linear)

    @property
    def image_value_projection(self) -> fl.Linear:
        return self.palette_cross_attention.layer(("Distribute", 2, "Linear"), fl.Linear)

    @property
    def weights(self) -> list[Tensor]:
        lst = [
            self.image_key_projection.weight,
            self.image_value_projection.weight
        ]
        if self.image_key_projection.bias is not None:
            lst.append(self.image_key_projection.bias)
        
        if self.image_value_projection.bias is not None:
            lst.append(self.image_value_projection.bias)
        
        return lst

class SD1PaletteAdapter(fl.Chain, Adapter[TSDNet]):
    # Prevent PyTorch module registration
    _palette_encoder: list[PaletteEncoder]

    def __init__(
        self,
        target: TSDNet,
        palette_encoder: PaletteEncoder,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self._palette_encoder = [palette_encoder]

        self.sub_adapters: list[PaletteCrossAttentionAdapter] = [
            PaletteCrossAttentionAdapter(
                target=cross_attn, scale=scale, embedding_dim=palette_encoder.embedding_dim
            )
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]
        
        if weights is not None:
            raise NotImplementedError("Loading weights is not implemented yet")
            # palette_state_dict: dict[str, Tensor] = {
            #     k.removeprefix("palette_encoder."): v for k, v in weights.items() if k.startswith("palette_encoder.")
            # }
            # self._palette_encoder[0].load_state_dict(palette_state_dict)
            
            # for i, cross_attn in enumerate(self.sub_adapters):
            #     # cross_attention_weights: list[Tensor] = []
                
            #     ## Tmp code
            #     index = i*2
            #     index2 = index + 1
            #     cross_attn.load_weights(
            #         weights[f"palette_adapter.{index:03d}"],
            #         weights[f"palette_adapter.{index2:03d}"],
            #     )
                
                # prefix = f"palette_adapter.{i:03d}."
                # for k, v in weights.items():
                #     if not k.startswith(prefix):
                #         continue
                #     cross_attention_weights.append(v)

                # assert len(cross_attention_weights) == 2
                # cross_attn.load_weights(*cross_attention_weights)
    @property
    def weights(self) -> List[Tensor]:
        weights: List[Tensor] = []
        for adapter in self.sub_adapters:
            weights.extend(adapter.weights)
        return weights

    def zero_init(self) -> None:
        weights = self.weights
        for weight in weights:
            init.zeros_(weight) # type: ignore

    def inject(self, parent: fl.Chain | None = None) -> "SD1PaletteAdapter[Any]":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()

    def set_scale(self, scale: float) -> None:
        for cross_attn in self.sub_adapters:
            cross_attn.scale = scale

    def set_palette_embedding(self, palette_embedding: Tensor) -> None:
        self.set_context("ip_adapter", {"palette_embedding": palette_embedding})

    @property
    def palette_encoder(self) -> PaletteEncoder:
        return self._palette_encoder[0]
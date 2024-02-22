from typing import Any, Iterator, List, TypeVar

from geomloss import SamplesLoss  # type: ignore
from PIL import Image
from torch import (
    Tensor,
    cat,
    device as Device,
    dtype as DType,
    flatten,
    float32,
    histogram,
    histogramdd,
    min,
    rand,
    sort,
    stack,
    zeros_like,
)
from torch.nn import L1Loss, Parameter, init
from torch.nn.functional import kl_div as _kl_div, mse_loss as _mse_loss

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.fluxion.utils import images_to_tensor
from refiners.foundationals.clip.common import FeedForward, PositionalEncoder
from refiners.foundationals.clip.image_encoder import ClassToken, TransformerLayer
from refiners.foundationals.clip.common import PositionalEncoder, FeedForward
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from ip_adapter_palette.layers import ViT3dEmbeddings

def images_to_histo_channels(images: List[Image.Image], color_bits: int = 8) -> List[Tensor]:
    img_tensor = images_to_tensor(images)
    sorted_channels = tensor_to_sorted_channels(img_tensor)
    return sorted_channels_to_histo_channels(sorted_channels, color_bits=color_bits)
    

def tensor_to_sorted_channels(image: Tensor, color_bits: int = 8, extended : bool = False) -> List[Tensor]:
    sorted_channels: List[Tensor] = []
    channels: List[Tensor] = image.split(1, dim=1) # type: ignore
    if extended:
        [red, green, blue] = channels # type: ignore
        channels = [
            red,
            green, 
            blue,
            (red+green)/2,
            (red+blue)/2,
            (green+blue)/2
        ]
    
    for channel in channels: # type: ignore
        # We extract RGB curves
        sorted_channel, _ = sort(flatten(channel, 1)) # type: ignore
        sorted_channels.append(sorted_channel)
    return sorted_channels

def sorted_channels_to_histo_channels(sorted_channels: List[Tensor], color_bits: int = 8) -> List[Tensor]:
    histos: List[Tensor] = []
    for channel in sorted_channels:
        histograms: List[Tensor] = []
        for i in range(channel.shape[0]):
            elem = channel[i]
            histo, _ = histogram(
                        elem.to(dtype=float32).cpu(),
                        bins=2**color_bits,
                        range= (0.0,1.0)
                    )
            histograms.append(histo/elem.numel())
        histo = stack(histograms)
        histos.append(histo.to(device = channel.device, dtype=channel.dtype))
    return histos

def histogram_to_histo_channels(histogram: Tensor) -> List[Tensor]:
    red = histogram.sum(dim=(2,3))
    green = histogram.sum(dim=(1,3))
    blue = histogram.sum(dim=(1,2))
    
    return [red, green, blue]

def expand_channels(rgb_channels : list[Tensor]) -> List[Tensor]:
    if len(rgb_channels) != 3:
        raise ValueError("3 channels expected")
    [red, green, blue] = rgb_channels
    return [
        red,
        green,
        blue
    ]
    

class ColorLoss(fl.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = L1Loss()

    def forward(self, actual: Tensor, expected: Tensor) -> Tensor:
        assert actual.shape == expected.shape, f"Shapes should match {actual.shape}/{expected.shape}"
        assert actual.shape[1] == 3, f"3 channels (R,G,B) image expected"
        actual_channels = tensor_to_sorted_channels(actual, extended=True)
        expected_channels = tensor_to_sorted_channels(expected, extended=True)
        
        actual_channels_tensor = cat([
            channel.unsqueeze(1) for channel in actual_channels
        ], dim=1)
        
        expected_channels_tensor = cat([
            channel.unsqueeze(1) for channel in expected_channels
        ], dim=1)
        
        return self.l1_loss(actual_channels_tensor, expected_channels_tensor)
    
    def image_vs_histo(self, image: Tensor, histo: Tensor, color_bits: int) -> List[Tensor]:
        actual_channels = tensor_to_sorted_channels(image, extended=False)
        histo_channels = histogram_to_histo_channels(histo)
        image_histo_channels = sorted_channels_to_histo_channels(actual_channels, color_bits=color_bits)
        return [
            self.l1_loss(histo_chan, image_histo_channels[i]) for i, histo_chan in enumerate(histo_channels)
        ]

def sample_points(histogram: Tensor, num_samples: int = 1000, color_bits : int= 4) -> Tensor:
    histo = histogram.reshape(histogram.shape[0], -1)
    num_bins = histo.shape[1]
    cube_size = histogram.shape[1]
    cdf = histo.cumsum(dim=1)
    cdf = cdf / cdf[:, -1].unsqueeze(1)
    uniform_samples = rand(num_samples, num_bins, device=histogram.device, dtype=histogram.dtype)
    onedim = (uniform_samples.unsqueeze(0) < cdf.unsqueeze(1)).int().argmax(dim=2)
    cubed_dim = stack([onedim // cube_size // cube_size, (onedim // cube_size) % cube_size, onedim % cube_size], -1)
    return cubed_dim.float()

from math import sqrt


class HistogramDistance(fl.Chain):
    def __init__(
        self,
        color_bits: int = 8,
    ) -> None:
        self.color_bits = color_bits
        super().__init__(fl.Lambda(func=self.kl_div))

    def mse(self, x: Tensor, y: Tensor) -> Tensor:
        return _mse_loss(x, y)
    
    def emd(self, x: Tensor, y: Tensor) -> Tensor:
        color_size = x.shape[1]
        emd_loss = SamplesLoss("sinkhorn", p=2, blur=1.0, diameter=sqrt(3*color_size*color_size))

        s_y = sample_points(x)
        s_x = sample_points(y)
        emd = emd_loss(s_x, s_y)
        print(f"EMD: {emd.mean()}, shape: {emd.shape}")
        return emd.mean()
    
    def correlation(self, x: Tensor, y: Tensor) -> Tensor:
        n = (2 ** self.color_bits) ** 3
        
        centered_x = x - 1/n
        centered_y = y - 1/n
                
        denom = ((centered_x*centered_x).sum() * (centered_y*centered_y).sum()).sqrt()
        return (centered_x*centered_y).sum()/denom
    
    def chi_square(self, x: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
        return (2*((x - y)**2)/(x + y + eps)).sum()/x.shape[0]

    def intersection(self, x: Tensor, y: Tensor) -> Tensor:
        return min(stack([x,y]), dim=0)[0].sum()/x.shape[0]
    
    def hellinger(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(x.shape[0], -1)
        
        base = x.sqrt() - y.sqrt()
        dist = (base * base).sum(dim = 1).sqrt()
        return dist.mean()
    
    def kl_div(self, actual: Tensor, expected: Tensor) -> Tensor:
        return _kl_div(actual, expected)
    
    def metrics_log(self, log: Tensor, y: Tensor) -> dict[str, Tensor]:
        x = log.exp()
        
        return {
            "mse": self.mse(x, y),
            "correlation": self.correlation(x, y),
            "chi_square": self.chi_square(x, y),
            "intersection": self.intersection(x, y),
            "hellinger": self.hellinger(x, y),
            "kl_div": self.kl_div(log, y),
            "emd": self.emd(x, y)
        }
    def metrics(self, x: Tensor, y: Tensor, eps: float = 1e-7) -> dict[str, Tensor]:
        
        return {
            "mse": self.mse(x, y),
            "correlation": self.correlation(x, y),
            "chi_square": self.chi_square(x, y),
            "intersection": self.intersection(x, y),
            "hellinger": self.hellinger(x, y),
            "kl_div": self.kl_div((x+eps).log(), y),
            "emd": self.emd(x, y)
        }
class HistogramExtractor(fl.Chain):
    def __init__(
        self,
        color_bits: int = 8,
    ) -> None:
        self.color_bits = color_bits
        self.color_size = 2**color_bits
        super().__init__(fl.Permute(0, 2, 3, 1), fl.Lambda(func=self.histogramdd))

    def histogramdd(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        num_pixels = x.shape[1] * x.shape[2]
        histograms: List[Tensor] = []
        device = x.device
        dtype = x.dtype
        # x is a [0, 1] normalized image
        x = x * (self.color_size - 1)

        for i in range(batch_size):
            hist_dd = histogramdd(
                x[i].to(dtype=float32).cpu(),
                bins=2**self.color_bits,
                range=[
                    0,
                    2**self.color_bits,
                    0,
                    2**self.color_bits,
                    0,
                    2**self.color_bits,
                ],
            )
            hist = hist_dd.hist / num_pixels
            histograms.append(hist)

        return stack(histograms).to(device, dtype)
    
    def images_to_histograms(self, images: List[Image.Image], device: Device | None = None, dtype : DType | None = None) -> Tensor:
        return self(images_to_tensor(images, device=device, dtype = dtype))


class HistogramEncoder(fl.Chain):
    def __init__(
        self,
        color_bits: int = 8,
        embedding_dim: int = 768,
        patch_size: int = 8,
        num_layers: int = 3,
        num_attention_heads: int = 3,
        feedforward_dim: int = 512,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.color_bits = color_bits
        cube_size = 2**color_bits
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        super().__init__(
            fl.Reshape(1, cube_size, cube_size, cube_size),
            ViT3dEmbeddings(
                cube_size=cube_size, embedding_dim=embedding_dim, patch_size=patch_size, device=device, dtype=dtype
            ),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
            fl.Chain(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    num_attention_heads=num_attention_heads,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
        )

    def compute_histogram_embedding(
        self,
        x: Tensor,
        negative_histogram: None | Tensor = None,
    ) -> Tensor:
        conditional_embedding = self(x)
        if x == negative_histogram:
            return cat(tensors=(conditional_embedding, conditional_embedding), dim=0)

        if negative_histogram is None:
            # a uniform palette with all the colors at the same frequency
            numel: int = x.numel()
            if numel == 0:
                raise ValueError("Cannot compute histogram embedding for empty tensor")
            negative_histogram = (zeros_like(x) + 1.0) * 1 / numel

        negative_embedding = self(negative_histogram)
        return cat(tensors=(negative_embedding, conditional_embedding), dim=0)


class HistogramCrossAttention(fl.Chain):
    def __init__(self, text_cross_attention: fl.Attention, embedding_dim: int = 768, scale: float = 1.0) -> None:
        self._scale = scale
        super().__init__(
            fl.Distribute(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="histogram_embedding"),
                    fl.Linear(
                        in_features=embedding_dim,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="histogram_embedding"),
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


class HistogramCrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(
        self,
        target: fl.Attention,
        scale: float = 1.0,
        embedding_dim: int = 768
    ) -> None:
        self._scale = scale
        with self.setup_adapter(target):
            clone = target.structural_copy()
            scaled_dot_product = clone.ensure_find(ScaledDotProductAttention)
            histogram_cross_attention = HistogramCrossAttention(
                text_cross_attention=clone,
                embedding_dim=embedding_dim,
                scale=self.scale,
            )
            clone.replace(
                old_module=scaled_dot_product,
                new_module=fl.Sum(
                    scaled_dot_product,
                    histogram_cross_attention,
                ),
            )
            super().__init__(
                clone,
            )

    @property
    def histogram_cross_attention(self) -> HistogramCrossAttention:
        return self.ensure_find(HistogramCrossAttention)

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.histogram_cross_attention.scale = value

    @property
    def weights(self) -> Iterator[Parameter]:
        return self.histogram_cross_attention.parameters()




TSDNet = TypeVar("TSDNet", bound="SD1UNet | SDXLUNet")


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


class HistogramProjection(fl.Chain):
    def __init__(
        self,
        in_features: int = 64,
        embedding_dim: int = 768,
        num_tokens: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 2048,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        
        features = embedding_dim * num_tokens
        
        super().__init__(
            fl.Linear(
                in_features=in_features,
                out_features=features,
                device=device,
                dtype=dtype,
            ),
            *(
                fl.Chain(
                    fl.LayerNorm(
                        normalized_shape=features,
                        eps=layer_norm_eps,
                        device=device,
                        dtype=dtype,
                    ),
                    FeedForward(
                        embedding_dim=features,
                        feedforward_dim=feedforward_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                for _ in range(num_layers)
            ),
            fl.Reshape(num_tokens, embedding_dim),
            fl.LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
        )

class SD1HistogramAdapter(fl.Chain, Adapter[TSDNet]):
    # Prevent PyTorch module registration
    _histogram_encoder: list[HistogramEncoder]

    def __init__(
        self,
        target: TSDNet,
        embedding_dim: int = 768,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self.sub_adapters: list[HistogramCrossAttentionAdapter] = [
            HistogramCrossAttentionAdapter(target=cross_attn, scale=scale, embedding_dim=embedding_dim)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

    @property
    def weights(self) -> List[Tensor]:
        weights: List[Tensor] = []
        for adapter in self.sub_adapters:
            weights += adapter.weights
        return weights

    def zero_init(self) -> None:
        weights = self.weights
        for weight in weights:
            init.zeros_(weight)

    def inject(self, parent: fl.Chain | None = None) -> "SD1HistogramAdapter[Any]":
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

    def set_histogram_embedding(self, histogram_embedding: Tensor) -> None:
        self.set_context("ip_adapter", {"histogram_embedding": histogram_embedding})

from typing import Any, List, TypeVar, cast

from jaxtyping import Float
from torch import Tensor, cat, device as Device, dtype as DType, float32, ones, tensor, zeros
from torch.nn.functional import pad
from torch.nn import init

import refiners.fluxion.layers as fl
from functools import cached_property
from ip_adapter_palette.types import Color, Palette, PaletteCluster
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.fluxion.layers.basics import Parameter
from refiners.foundationals.clip.common import FeedForward, PositionalEncoder
from refiners.foundationals.clip.text_encoder import TransformerLayer
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet



class TransformerEncoder(fl.Chain):

    def __init__(
        self,
        embedding_dim: int = 768,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        feedforward_dim: int = 20,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        # self._lda = [lda]
        self.embedding_dim = embedding_dim
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

class MLPEncoder(fl.Chain):

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
                    fl.Chain(
                        fl.Linear(in_features=embedding_dim, out_features=feedforward_dim, bias=False, device=device, dtype=dtype),
                        fl.GeLU(),
                        fl.Linear(in_features=feedforward_dim, out_features=embedding_dim, bias=False, device=device, dtype=dtype),
                    )
                )
                for _ in range(num_layers)
            )
        )


class CrossAttentionChangeLength(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        output_len: int,
        num_heads: int = 1,
        inner_dim: int | None = None,
        input_dim: int | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Parallel(
                fl.Parameter(output_len, embedding_dim),
                fl.Identity(),
                fl.Identity(),
            ),
            fl.Attention(
                embedding_dim=embedding_dim,
                key_embedding_dim=input_dim,
                value_embedding_dim=input_dim,
                inner_dim=inner_dim,
                num_heads=num_heads,
                use_bias=False,
                device=device,
                dtype=dtype,
            )
        )


class GenericEncoder(fl.Chain):
    def __init__(
        self,
        input_dim: int = 768,
        embedding_dim: int = 768,
        output_len: int | None = None,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        feedforward_dim: int = 20,
        layer_norm_eps: float = 1e-5,
        mode: str = 'transformer',
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        empty = True
        if output_len is not None:
            encoder_head = fl.Chain(
                MLPEncoder(
                    embedding_dim=feedforward_dim,
                    num_layers=1,
                    feedforward_dim=feedforward_dim,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype
                ),
                CrossAttentionChangeLength(
                    embedding_dim=embedding_dim,
                    input_dim=feedforward_dim,
                    inner_dim=feedforward_dim,
                    output_len=output_len,
                    device=device,
                    dtype=dtype
                )
            )
            empty = False
        elif input_dim != embedding_dim:
            encoder_head = fl.Linear(in_features=input_dim, out_features=embedding_dim, bias=False, device=device, dtype=dtype)
            empty = False
        else: 
            encoder_head = fl.Identity()

        if mode == 'transformer' and num_layers > 0:
            empty = False
            encoder_body = TransformerEncoder(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                feedforward_dim=feedforward_dim,
                layer_norm_eps=layer_norm_eps,
                device=device,
                dtype=dtype
            )
        elif mode == 'mlp' and num_layers > 0:
            empty = False
            encoder_body = MLPEncoder(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                feedforward_dim=feedforward_dim,
                layer_norm_eps=layer_norm_eps,
                device=device,
                dtype=dtype
            )
        else: 
            encoder_body = fl.Identity()
        
        if empty:
            encoder_tail = fl.Identity()
        else:
            encoder_tail = fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype)

        super().__init__(
            encoder_head,
            encoder_body,
            encoder_tail
        )
    
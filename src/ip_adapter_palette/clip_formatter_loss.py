from typing import Literal
import refiners.fluxion.layers as fl 
from torch import dtype as DType, device as Device, Tensor, layer_norm, nn
from refiners.fluxion.layers import Chain, Parallel, ScaledDotProductAttention, Lambda, Chain, Sum
from refiners.fluxion.adapters.lora import LinearLora
from ip_adapter_palette.config import CLIPFormatterLossModes

class CrossAttentionLora(Chain):

    def __init__(
        self,
        embedding_dim: int,
        rank: int,
        num_heads: int = 1,
        mode: CLIPFormatterLossModes = "sum",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:

        assert (
            embedding_dim % num_heads == 0
        ), f"embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.heads_dim = embedding_dim // num_heads
        self.mode = mode

        if self.mode == "cross_attention":
            body = Chain(
                Parallel(
                    fl.Identity(),
                    fl.Chain(
                        fl.UseContext(context="ip_adapter", key="clip_palette_embedding"),
                        LinearLora(  # key projection
                            "key",
                            in_features=self.embedding_dim,
                            out_features=self.embedding_dim,
                            rank=rank,
                            device=device,
                            dtype=dtype,
                        ),
                    ),
                    fl.Chain(
                        fl.UseContext(context="ip_adapter", key="clip_palette_embedding"),
                        LinearLora(  # Value projection
                            "value",
                            in_features=self.embedding_dim,
                            out_features=self.embedding_dim,
                            rank=rank,
                            device=device,
                            dtype=dtype,
                        ),
                    ),
                ),
                ScaledDotProductAttention(
                    num_heads=num_heads,
                    is_causal=False,
                ),
                LinearLora(  # Value projection
                    "output",
                    in_features=self.embedding_dim,
                    out_features=self.embedding_dim,
                    rank=rank,
                    device=device,
                    dtype=dtype,
                ),
            )
        elif self.mode == "cross_attention_query":
           body = Chain(
                Parallel(
                    fl.Chain(
                        fl.UseContext(context="ip_adapter", key="clip_palette_embedding"),
                        LinearLora(  # Query projection
                            "query",
                            in_features=self.embedding_dim,
                            out_features=self.embedding_dim,
                            rank=rank,
                            device=device,
                            dtype=dtype,
                        ),
                    ),
                    fl.Identity(),
                    fl.Identity()
                ),
                ScaledDotProductAttention(
                    num_heads=num_heads,
                    is_causal=False,
                ),
                LinearLora(  # Value projection
                    "output",
                    in_features=self.embedding_dim,
                    out_features=self.embedding_dim,
                    rank=rank,
                    device=device,
                    dtype=dtype,
                ),
            )
        else:
            body = Chain(
                fl.Residual(
                    fl.UseContext(context="ip_adapter", key="clip_palette_embedding"),
                )
            )


        super().__init__(
            body,
            fl.LayerNorm(normalized_shape=self.embedding_dim, device=device, dtype=dtype),
        )


class CLIPFormatterLoss(fl.Module):
    def __init__(self, device: Device, dtype: DType, rank: int = 16, mode: CLIPFormatterLossModes = "sum", embedding_dim: int = 768, use_bias: bool = False) -> None:
        super().__init__()
        self.cross_attention = CrossAttentionLora(
            embedding_dim=embedding_dim,
            rank= rank,
            num_heads=1,
            mode=mode,
            device=device,
            dtype=dtype,
        )
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, palette_embedding: Tensor, image_embedding: Tensor, bw_image_embedding: Tensor) -> Tensor:
        self.cross_attention.set_context("ip_adapter", {"clip_palette_embedding":palette_embedding})
        
        return self.loss(self.cross_attention(bw_image_embedding), image_embedding)
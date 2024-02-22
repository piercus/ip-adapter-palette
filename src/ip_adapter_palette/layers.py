from torch import nn, device as Device, dtype as DType, Tensor, Size
from refiners.fluxion import layers as fl
from refiners.fluxion.layers import Parallel, Identity, Interpolate, Lambda, SetContext, UseContext, WeightedModule, Chain, Sum, Conv2d, GroupNorm, SiLU, SelfAttention2d, Downsample, Residual, Slicing, Upsample, SelfAttention
from refiners.fluxion.utils import pad
from typing import Callable
from refiners.foundationals.clip.image_encoder import ClassToken
from refiners.foundationals.clip.common import PositionalEncoder
from refiners.fluxion.context import Contexts
from jaxtyping import Float



class Conv3d(nn.Conv3d, WeightedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = (1, 1, 1),
        padding: int | tuple[int, int, int] | str = (0, 0, 0),
        groups: int = 1,
        use_bias: bool = True,
        dilation: int | tuple[int, int, int] = (1, 1, 1),
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            device,
            dtype,
        )
        self.use_bias = use_bias



class Downsample3d(Chain):
    def __init__(
        self,
        channels: int,
        scale_factor: int,
        padding: int = 0,
        register_shape: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        """Downsamples the input by the given scale factor.

        If register_shape is True, the input shape is registered in the context. It will throw an error if the context
        sampling is not set or if the context does not contain a list.
        """
        self.channels = channels
        self.in_channels = channels
        self.out_channels = channels
        self.scale_factor = scale_factor
        self.padding = padding
        super().__init__(
            Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=scale_factor,
                padding=padding,
                device=device,
                dtype=dtype,
            ),
        )
        if padding == 0:
            zero_pad: Callable[[Tensor], Tensor] = lambda x: pad(x, (0, 1, 0, 1, 0, 1))
            self.insert(0, Lambda(zero_pad))
        if register_shape:
            self.insert(0, SetContext(context="sampling", key="shapes", callback=self.register_shape))

    def register_shape(self, shapes: list[Size], x: Tensor) -> None:
        shapes.append(x.shape[2:])


class Upsample3d(Chain):
    def __init__(
        self,
        channels: int,
        upsample_factor: int | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        """Upsamples the input by the given scale factor.

        If upsample_factor is None, the input shape is taken from the context. It will throw an error if the context
        sampling is not set or if the context is empty (then you should use the dynamic version of Downsample).
        """
        self.channels = channels
        self.upsample_factor = upsample_factor
        super().__init__(
            Parallel(
                Identity(),
                (
                    Lambda(self._get_static_shape)
                    if upsample_factor is not None
                    else UseContext(context="sampling", key="shapes").compose(lambda x: x.pop())
                ),
            ),
            Interpolate(),
            Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
        )

    def _get_static_shape(self, x: Tensor) -> Size:
        assert self.upsample_factor is not None
        return Size([size * self.upsample_factor for size in x.shape[2:]])



class Patch3dEncoder(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 8,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_bias = use_bias
        super().__init__(
            Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.patch_size, self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size, self.patch_size),
                use_bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
        )


class SelfAttention3d(SelfAttention):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        use_bias: bool = True,
        is_causal: bool = False,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        self.channels = channels
        super().__init__(
            embedding_dim=channels,
            num_heads=num_heads,
            use_bias=use_bias,
            is_causal=is_causal,
            is_optimized=is_optimized,
            device=device,
            dtype=dtype,
        )
        self.insert(0, Lambda(self.tensor_3d_to_sequence))
        self.append(Lambda(self.sequence_to_tensor_3d))

    def init_context(self) -> Contexts:
        return {"reshape": {"height": None, "width": None, "depth": None}}

    def tensor_3d_to_sequence(
        self, x: Float[Tensor, "batch channels height width depth"]
    ) -> Float[Tensor, "batch height*width*depth channels"]:
        height, width, depth = x.shape[-3:]
        self.set_context(context="reshape", value={"height": height, "width": width, "depth": depth})
        return x.reshape(x.shape[0], x.shape[1], height * width * depth).transpose(1, 2)

    def sequence_to_tensor_3d(
        self, x: Float[Tensor, "batch sequence_length channels"]
    ) -> Float[Tensor, "batch channels height width depth"]:
        height, width, depth = self.use_context("reshape").values()
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width, depth)



class ViT3dEmbeddings(fl.Chain):
    def __init__(
        self,
        cube_size: int = 256,
        embedding_dim: int = 768,
        patch_size: int = 8,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.cube_size = cube_size
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        super().__init__(
            fl.Concatenate(
                ClassToken(embedding_dim, device=device, dtype=dtype),
                fl.Chain(
                    Patch3dEncoder(
                        in_channels=1,
                        out_channels=embedding_dim,
                        patch_size=patch_size,
                        use_bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.Reshape((cube_size // patch_size) ** 3, embedding_dim),
                ),
                dim=1,
            ),
            fl.Residual(
                PositionalEncoder(
                    max_sequence_length=(cube_size // patch_size) ** 3 + 1,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class Resnet(Sum):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        spatial_dims: int = 2,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if spatial_dims == 2:
            Conv = Conv2d
        elif spatial_dims == 3:
            Conv = Conv3d
        else:
            raise ValueError(f"Unsupported spatial dimension {spatial_dims}")
        
        shortcut = (
            Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else Identity()
        )
        super().__init__(
            shortcut,
            Chain(
                GroupNorm(channels=in_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                GroupNorm(channels=out_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class Encoder(Chain):
    def __init__(self, 
        spatial_dims: int = 2, 
        num_groups: int = 32, 
        resnet_sizes: list[int] = [128, 256, 512, 512, 512],
        input_channels: int = 3,
        n_down_samples: int = 3,
        latent_dim: int = 8,
        slide_end: int = 4,
        device: Device | str | None = None, 
        dtype: DType | None = None,        
    ) -> None:
        if spatial_dims == 2:
            Conv = Conv2d
            SelfAttention = SelfAttention2d
            Dsample = Downsample
        elif spatial_dims == 3:
            Conv = Conv3d
            SelfAttention = SelfAttention3d
            Dsample = Downsample3d
        else:
            raise ValueError(f"Unsupported spatial dimension {spatial_dims}")
        
        resnet_layers: list[Chain] = [
            Chain(
                [
                    Resnet(
                        in_channels=resnet_sizes[i - 1] if i > 0 else resnet_sizes[0],
                        out_channels=resnet_sizes[i],
                        num_groups=num_groups,
                        spatial_dims=spatial_dims,
                        device=device,
                        dtype=dtype,
                    ),
                    Resnet(
                        in_channels=resnet_sizes[i],
                        num_groups=num_groups,
                        out_channels=resnet_sizes[i],
                        spatial_dims=spatial_dims,                        
                        device=device,
                        dtype=dtype,
                    ),
                ]
            )
            for i in range(len(resnet_sizes))
        ]
        for _, layer in zip(range(n_down_samples), resnet_layers):
            channels: int = layer[-1].out_channels  # type: ignore
            layer.append(Dsample(channels=channels, scale_factor=2, device=device, dtype=dtype))

        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[-1], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
            SelfAttention(channels=resnet_sizes[-1], device=device, dtype=dtype),
        )
        resnet_layers[-1].insert_after_type(Resnet, attention_layer)
        super().__init__(
            Conv(
                in_channels=input_channels,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=resnet_sizes[-1],
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Chain(
                Conv(in_channels=latent_dim, out_channels=latent_dim, kernel_size=1, device=device, dtype=dtype),
                Slicing(dim=1, end=slide_end),
            ),
        )

    def init_context(self) -> Contexts:
        return {"sampling": {"shapes": []}}


class Decoder(Chain):
    def __init__(self, 
        spatial_dims: int = 2, 
        num_groups: int = 32, 
        resnet_sizes: list[int] = [128, 256, 512, 512, 512],
        output_channels: int = 3,
        latent_dim: int = 4,
        n_up_samples: int = 3,
        device: Device | str | None = None, 
        dtype: DType | None = None,     
    ) -> None:    
                     
        if spatial_dims == 2:
            Conv = Conv2d
            SelfAttention = SelfAttention2d
            Usample = Upsample
        elif spatial_dims == 3:
            Conv = Conv3d
            SelfAttention = SelfAttention3d
            Usample = Upsample3d
        else:
            raise ValueError(f"Unsupported spatial dimension {spatial_dims}")
         
        resnet_sizes = resnet_sizes[::-1]
        
        resnet_layers: list[Chain] = [
            (
                Chain(
                    [
                        Resnet(
                            in_channels=resnet_sizes[i - 1],
                            out_channels=resnet_sizes[i],
                            num_groups=num_groups,
                            spatial_dims=spatial_dims,
                            device=device,
                            dtype=dtype,
                        )
                        if i > 0
                        else Identity(),
                        Resnet(
                            in_channels=resnet_sizes[i],
                            out_channels=resnet_sizes[i],
                            num_groups=num_groups,
                            spatial_dims=spatial_dims,
                            device=device,
                            dtype=dtype,
                        ),
                        Resnet(
                            in_channels=resnet_sizes[i],
                            out_channels=resnet_sizes[i],
                            num_groups=num_groups,
                            spatial_dims=spatial_dims,
                            device=device,
                            dtype=dtype,
                        ),
                    ]
                )
            )
            for i in range(len(resnet_sizes))
        ]
        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[0], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
            SelfAttention(channels=resnet_sizes[0], device=device, dtype=dtype),
        )
        resnet_layers[0].insert(1, attention_layer)
        if n_up_samples > len(resnet_layers) - 1:
            raise ValueError(
                f"Number of up-samples ({n_up_samples}) must be less than or equal to the number of resnet layers - 1 ({len(resnet_layers)})"
            )
        for _, layer in zip(range(n_up_samples), resnet_layers[1:]):
            channels: int = layer[-1].out_channels # type: ignore
            layer.insert(-1, Usample(channels=channels, upsample_factor=2, device=device, dtype=dtype))
        super().__init__(
            Conv(
                in_channels=latent_dim, out_channels=latent_dim, kernel_size=1, device=device, dtype=dtype
            ),
            Conv(
                in_channels=latent_dim,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=resnet_sizes[-1],
                    out_channels=output_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )

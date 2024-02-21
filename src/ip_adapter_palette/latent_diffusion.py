from abc import ABC
from functools import cached_property

import torch
from PIL import Image
from torch import device as Device, dtype as DType

from ip_adapter_palette.config import LatentDiffusionConfig, SDModelConfig
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.auto_encoder import (
    LatentDiffusionAutoencoder,
)
from refiners.foundationals.latent_diffusion.solvers import DDPM
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import (
    StableDiffusion_1,
)
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.training_utils import Trainer, register_model
from refiners.training_utils.config import BaseConfig


class BaseLatentDiffusionConfig(BaseConfig, ABC):
    latent_diffusion: LatentDiffusionConfig


def resize_image(
    image: Image.Image, min_size: int = 512, max_size: int = 576
) -> Image.Image:
    """
    Resize an image so that the smallest side is between `min_size` and `max_size`.
    """
    image_min_size = min(image.size)
    if image_min_size > max_size:
        if image_min_size == image.size[0]:
            image = image.resize(
                size=(max_size, int(max_size * image.size[1] / image.size[0]))
            )
        else:
            image = image.resize(
                size=(int(max_size * image.size[0] / image.size[1]), max_size)
            )
    if image_min_size < min_size:
        if image_min_size == image.size[0]:
            image = image.resize(
                size=(min_size, int(min_size * image.size[1] / image.size[0]))
            )
        else:
            image = image.resize(
                size=(int(min_size * image.size[0] / image.size[1]), min_size)
            )
    return image


def sample_noise(
    size: tuple[int, ...],
    /,
    offset_noise: float = 0.1,
    generator: torch.Generator | None = None,
    device: Device | str | None = None,
    dtype: DType | None = None,
) -> torch.Tensor:
    """Sample noise from a normal distribution.

    If `offset_noise` is more than 0, the noise will be offset by a small amount. It allows the model to generate
    images with a wider range of contrast https://www.crosslabs.org/blog/diffusion-with-offset-noise.
    """
    noise = torch.randn(*size, generator=generator, device=device, dtype=dtype)
    return noise + offset_noise * torch.randn(
        *size[:2], 1, 1, generator=generator, device=device, dtype=dtype
    )



class SD1TrainerMixin(ABC):
    config: BaseLatentDiffusionConfig

    @register_model()
    def sd(self, config: SDModelConfig) -> StableDiffusion_1:
        assert isinstance(self, Trainer), "This mixin can only be used with a Trainer"
        sd = StableDiffusion_1(device=self.device, dtype=self.dtype)
        sd.unet.load_from_safetensors(config.unet)
        sd.clip_text_encoder.load_from_safetensors(config.text_encoder)
        sd.lda.load_from_safetensors(config.lda)
        sd.unet.requires_grad_(False)
        sd.clip_text_encoder.requires_grad_(False)
        sd.lda.requires_grad_(False)
        return sd

    @cached_property
    def lda(self) -> LatentDiffusionAutoencoder:
        return self.sd.lda

    @cached_property
    def text_encoder(self) -> CLIPTextEncoderL:
        return self.sd.clip_text_encoder

    @cached_property
    def unet(self) -> SD1UNet:
        return self.sd.unet

    @cached_property
    def ddpm_scheduler(self) -> DDPM:
        assert isinstance(self, Trainer), "This mixin can only be used with a Trainer"
        return DDPM(
            num_inference_steps=1000,
            device=self.device,
        )

    def sample_noise(
        self,
        size: tuple[int, ...],
        /,
        offset_noise: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noise from a normal distribution.

        If `offset_noise` is more than 0, the noise will be offset by a small amount. It allows the model to generate
        images with a wider range of contrast https://www.crosslabs.org/blog/diffusion-with-offset-noise.
        """
        assert isinstance(self, Trainer), "This mixin can only be used with a Trainer"
        return sample_noise(size, offset_noise, generator, self.device, self.dtype)

    def sample_timestep(self, batch_size: int, /) -> torch.Tensor:
        """Sample a timestep from a uniform distribution."""
        assert isinstance(self, Trainer), "This mixin can only be used with a Trainer"
        random_steps = torch.randint(0, 1000, (batch_size,))
        self.random_steps = random_steps
        return self.ddpm_scheduler.timesteps[random_steps]
    

    def add_noise_to_latents(
        self, latents: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to latents."""
        return torch.cat(
            [
                self.ddpm_scheduler.add_noise(
                    latents[i : i + 1], noise[i : i + 1], int(self.random_steps[i].item())
                )
                for i in range(latents.shape[0])
            ],
            dim=0,
        )

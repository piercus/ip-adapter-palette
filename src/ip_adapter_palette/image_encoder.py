
import refiners.fluxion.layers as fl
from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
from torch import device as Device, dtype as DType, Tensor, cat, no_grad, zeros
from refiners.fluxion.utils import image_to_tensor, normalize, load_from_safetensors
from PIL import Image
from functools import cached_property
from refiners.foundationals.latent_diffusion.image_prompt import PerceiverResampler

class ImageEncoder(fl.Module):

    # Prevent PyTorch module registration
    _clip_image_encoder: list[CLIPImageEncoderH]
    _grid_image_encoder: list[CLIPImageEncoderH]
    _image_proj: list[fl.Module]
    
    def __init__(
            self,
            filepath: str = 'weights/models/CLIPImageEncoderH.safetensors',
            image_proj_filepath: str = 'weights/models/IPAdapter.safetensors',
            clip_text_embedding_dim: int = 768,
            device: Device | str | None = None,
            dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self._clip_image_encoder = [CLIPImageEncoderH(
            device=device,
            dtype=dtype,
        )]
        self.clip_image_encoder.requires_grad_(False)
        self.clip_image_encoder.load_from_safetensors(filepath)
        self._grid_image_encoder = [self.convert_to_grid_features(self.clip_image_encoder)]
        
        image_proj = PerceiverResampler(
            latents_dim=clip_text_embedding_dim,
            num_attention_layers=4,
            num_attention_heads=12,
            head_dim=64,
            num_tokens=16,
            input_dim=self.clip_image_encoder.embedding_dim,
            output_dim=clip_text_embedding_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._image_proj = [image_proj]
        
        image_proj_weights = load_from_safetensors(image_proj_filepath)

        image_proj_state_dict: dict[str, Tensor] = {
            k.removeprefix("image_proj."): v for k, v in image_proj_weights.items() if k.startswith("image_proj.")
        }
        self.image_proj.load_state_dict(image_proj_state_dict)
        self.image_proj.requires_grad_(False)

    def forward(self, images: list[Image.Image]) -> Tensor:
        img_tensor = cat([self.preprocess_image(image) for image in images])
        return self.from_tensor(img_tensor)
    
    def from_tensor(self, img_tensor: Tensor) -> Tensor:
        encoded = self.grid_image_encoder(img_tensor)
        return self.image_proj(encoded)
    
    @property
    def clip_image_encoder(self) -> CLIPImageEncoderH:
        """The CLIP image encoder of the adapter."""
        return self._clip_image_encoder[0]

    @property
    def grid_image_encoder(self) -> CLIPImageEncoderH:
        assert hasattr(self, "_grid_image_encoder")
        return self._grid_image_encoder[0]

    @property
    def image_proj(self) -> fl.Module:
        return self._image_proj[0]
    
    @staticmethod
    def convert_to_grid_features(clip_image_encoder: CLIPImageEncoderH) -> CLIPImageEncoderH:
        encoder_clone = clip_image_encoder.structural_copy()
        assert isinstance(encoder_clone[-1], fl.Linear)  # final proj
        assert isinstance(encoder_clone[-2], fl.LayerNorm)  # final normalization
        assert isinstance(encoder_clone[-3], fl.Lambda)  # pooling (classif token)
        for _ in range(3):
            encoder_clone.pop()
        transformer_layers = encoder_clone[-1]
        assert isinstance(transformer_layers, fl.Chain) and len(transformer_layers) == 32
        transformer_layers.pop()
        return encoder_clone
    
    def to(self, device: Device | str | None = None, dtype: DType | None = None) -> "ImageEncoder":
        self.device = device if device is not None else self.device
        self.dtype = dtype if dtype is not None else self.dtype
        self.clip_image_encoder.to(device=device, dtype=dtype)
        self.image_proj.to(device=device, dtype=dtype)
        return super().to(device=device, dtype=dtype)

    def preprocess_image(
        self,
        image: Image.Image,
        size: tuple[int, int] = (224, 224),
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> Tensor:
        """Preprocess the image.

        Note:
            The default mean and std are parameters from
            https://github.com/openai/CLIP

        Args:
            image: The image to preprocess.
            size: The size to resize the image to.
            mean: The mean to use for normalization.
            std: The standard deviation to use for normalization.
        """
        return normalize(
            image_to_tensor(image.resize(size), device=self.device, dtype=self.dtype),
            mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
            std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
        )
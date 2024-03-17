from refiners.fluxion.utils import image_to_tensor, normalize, load_from_safetensors
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

def preprocess_image(
    image: Image.Image,
    size: tuple[int, int] = (224, 224),
    mean: list[float] | None = None,
    std: list[float] | None = None,
    device: Device | str | None = None,
    dtype: DType | None = None,
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
        image_to_tensor(image.resize(size), device=device, dtype=dtype),
        mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
        std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
    )
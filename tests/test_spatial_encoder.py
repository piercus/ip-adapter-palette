import torch

from ip_adapter_palette.spatial_encoder import SpatialTokenizer, Encoder
from refiners.fluxion.utils import image_to_tensor, tensor_to_image
from PIL import Image
import numpy as np

def test_spatial_tokenizer() -> None:

    image = Image.open("tests/fixtures/photo-1439246854758-f686a415d9da.jpeg").resize((512, 512))

    tokenizer = SpatialTokenizer()

    tokens = tokenizer([image])
    assert tokens.shape == (1, 3, 8, 8)

def test_encoder() -> None:
    image = Image.open("tests/fixtures/photo-1439246854758-f686a415d9da.jpeg").resize((512, 512))

    encoder = Encoder()
    encoded = encoder.images_to_latents([image])

    assert encoded.shape == (1, 65, 768)
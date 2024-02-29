from tracemalloc import start
import torch

from ip_adapter_palette.pixel_sampling import Sampler, Encoder
from refiners.fluxion.utils import image_to_tensor, tensor_to_image
from PIL import Image
import numpy as np
import time
def test_sampler() -> None:

    image = Image.open("tests/fixtures/photo-1439246854758-f686a415d9da.jpeg").resize((512, 512))

    sampler = Sampler()
    start = time.time()
    sample = sampler([image])
    print("Time to sample", time.time()-start)
    assert sample.shape == (1, 2048, 4)

def test_encoder() -> None:
    image = Image.open("tests/fixtures/photo-1439246854758-f686a415d9da.jpeg").resize((512, 512))

    encoder = Encoder(embedding_dim=768)
    encoded = encoder.images_to_latents([image])

    assert encoded.shape == (1, 2048, 768)
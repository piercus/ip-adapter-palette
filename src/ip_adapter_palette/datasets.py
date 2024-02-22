from refiners.training_utils.trainer import Batch
from torch.utils.data import Dataset
from typing import Any, Dict, cast, TypedDict
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig, load_hf_dataset, HuggingfaceDataset
from ip_adapter_palette.types import BatchInput
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DownloadManager, Image as DatasetImage # type: ignore
import os
from torch import Module as TorchModule, save, load, Tensor
from PIL import Image
from pathlib import Path
from loguru import logger
from refiners.foundationals.latent_diffusion import LatentDiffusionAutoencoder
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
class HfItem(TypedDict):
    photo_id: str
    image: Image.Image
    caption: str

class EmbeddingsItem(TypedDict):
    latents: Tensor
    text_embedding: Tensor

def resize_image(image: Image.Image, min_size: int = 512, max_size: int = 576) -> Image.Image:
    image_min_size = min(image.size)
    if image_min_size > max_size:
        if image_min_size == image.size[0]:
            image = image.resize(size=(max_size, int(max_size * image.size[1] / image.size[0])))
        else:
            image = image.resize(size=(int(max_size * image.size[0] / image.size[1]), max_size))
    if image_min_size < min_size:
        if image_min_size == image.size[0]:
            image = image.resize(size=(min_size, int(min_size * image.size[1] / image.size[0])))
        else:
            image = image.resize(size=(int(min_size * image.size[0] / image.size[1]), min_size))
    return image

class ResizeImage(TorchModule):
    def __init__(self, size: int = 512) -> None:
        super().__init__()
        self.size = size

    def forward(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return resize_image(image, self.size)

def build_hf_dataset(config: HuggingfaceDatasetConfig) -> HuggingfaceDataset[HfItem]:
        hf_dataset = load_hf_dataset(
            config.hf_repo,
            config.revision,
            config.split,
            config.use_verification
        )
        def download_image(url: str | list[str], dl_manager: DownloadManager):
            img = dl_manager.download(url)
            img = cast(Image.Image, img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = resize_image(img, config.resize_image_min_size, config.resize_image_max_size)
            return {"image": img}

        hf_dataset = hf_dataset.map( # type: ignore
            function=download_image,
            input_columns=["photo_image_url"],
            fn_kwargs={
                "dl_manager": DownloadManager(),
            },
            batched=True,
            num_proc=os.cpu_count()
        )
        return hf_dataset


def hf_item_to_batch_input(hf_item: HfItem, db_index: int, folder: Path) -> BatchInput:
    embeddings = load_embeddings_from_hf(folder, hf_item)
    return BatchInput(
        db_indexes = [db_index], 
        source_prompts = [hf_item["caption"]],
        source_images= [hf_item["image"]],
        latents = embeddings["latents"],
        text_embeddings = embeddings["text_embedding"]
    )

def save_hf_item_as_latents(
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        hf_item: HfItem, 
        force: bool,
        folder: Path
    ) -> None:

    photo_id = hf_item['photo_id']
    image = hf_item['image']
    caption = hf_item['caption']

    filename = folder / f"{photo_id}.pt"

    if filename.exists() and not force:
        logger.debug(f"Skipping {filename}. Already exists, change this behavior with --force.")
        return
    
    latents = lda.image_to_latents(image)
    text_embedding = text_encoder(caption)
    save({
            "latents": latents,
            "text_embedding": text_embedding
        },
        filename
    )

def load_embeddings_from_hf(folder: Path, hf_item: HfItem) -> EmbeddingsItem:
    photo_id = hf_item['photo_id']
    filename = folder / f"{photo_id}.pt"
    return cast(EmbeddingsItem, load(filename))


class EmbeddableDataset(Dataset[BatchInput]):
    def __init__(self,
        hf_dataset_config: HuggingfaceDatasetConfig,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        folder: Path
    ):
        self.hf_dataset = build_hf_dataset(hf_dataset_config)
        self.lda = lda
        self.text_encoder = text_encoder
        self.folder = folder
    
    def precompute_embeddings(self, force: bool = False) -> None:
        for hf_item in self.hf_dataset:
            save_hf_item_as_latents(self.lda, self.text_encoder, hf_item, force, self.folder)

class GridEvalDataset(EmbeddableDataset):
    def __init__(self,
        hf_dataset_config: HuggingfaceDatasetConfig,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        db_indexes: list[int],
        prompts: list[str],
        folder: Path
    ):
        super().__init__(hf_dataset_config, lda, text_encoder, folder)
        self.db_indexes = db_indexes
        self.prompts = prompts
    
    def __len__(self):
        return len(self.db_indexes) * len(self.prompts)
    
    def __getitem__(self, index: int) -> BatchInput:
        db_index = self.db_indexes[index // len(self.prompts)]
        prompt = self.prompts[index % len(self.prompts)]
        hf_item = self.hf_dataset[db_index]
        return hf_item_to_batch_input(hf_item, db_index, self.folder)

class ColorDataset(EmbeddableDataset):
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, index: int) -> BatchInput:
        hf_item = self.hf_dataset[index]
        return hf_item_to_batch_input(hf_item, index, self.folder)
        
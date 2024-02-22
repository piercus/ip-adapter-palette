from refiners.training_utils.trainer import Batch
from torch.utils.data import Dataset
from typing import Any, Dict, cast, TypedDict
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig, load_hf_dataset, HuggingfaceDataset
from ip_adapter_palette.types import BatchInput
from ip_adapter_palette.palette_adapter import PaletteExtractor
from ip_adapter_palette.histogram import HistogramExtractor

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DownloadManager, Image as DatasetImage # type: ignore
import os
from torch import save, load, Tensor
from PIL import Image
from pathlib import Path
from loguru import logger
from refiners.foundationals.latent_diffusion import LatentDiffusionAutoencoder
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.fluxion.utils import image_to_tensor
class HfItem(TypedDict):
    photo_id: str
    image: Image.Image
    caption: str

class EmbeddingsItem(TypedDict):
    latents: Tensor
    text_embedding: Tensor
    histogram: Tensor
    palette: Tensor

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


def hf_item_to_batch_input(hf_item: HfItem, folder: Path) -> BatchInput:
    return load_batch_from_hf(folder, hf_item)

def save_batch_as_latents(
        batch: BatchInput, 
        force: bool,
        folder: Path
    ) -> None:
    batch.id()
    filename = folder / f"{batch.id()}.pt"

    if filename.exists() and not force:
        logger.debug(f"Skipping {filename}. Already exists, change this behavior with --force.")
        return
    
    value_dict = {
        getattr(batch, key)
        for key in BatchInput.keys()
    }
    save(value_dict,
        filename
    )

def build_batch_from_hf_item(
        hf_item: HfItem,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        palette_extractor: PaletteExtractor,
        histogram_extractor: HistogramExtractor,
        force: bool,
        folder: Path,
        db_index: int
    ) -> BatchInput:
    
    return BatchInput(
        source_palettes= [palette_extractor(hf_item['image'])],
        source_prompts = hf_item['caption'],
        db_indexes = [db_index],
        photo_ids = [hf_item['photo_id']],
        source_images = image_to_tensor(hf_item['image']),
        source_text_embeddings = text_encoder(hf_item['caption']),
        source_latents = lda.image_to_latents(hf_item['image']),
        source_histograms = histogram_extractor(hf_item['image'])
    )

def load_batch_from_hf(folder: Path, hf_item: HfItem) -> BatchInput:
    photo_id = hf_item['photo_id']
    filename = folder / f"{photo_id}.pt"
    return BatchInput.load_file(filename)


class EmbeddableDataset(Dataset[BatchInput]):
    def __init__(self,
        hf_dataset_config: HuggingfaceDatasetConfig,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        palette_extractor: PaletteExtractor,
        histogram_extractor: HistogramExtractor,
        folder: Path
    ):
        self.hf_dataset = build_hf_dataset(hf_dataset_config)
        self.lda = lda
        self.text_encoder = text_encoder
        self.folder = folder
        self.palette_extractor = palette_extractor
        self.histogram_extractor = histogram_extractor
    
    def precompute_embeddings(self, force: bool = False) -> None:
        db_index = 0
        for hf_item in self.hf_dataset:
            build_batch_from_hf_item(
                hf_item,
                self.lda, 
                self.text_encoder, 
                self.palette_extractor,
                self.histogram_extractor,
                force, 
                self.folder,
                db_index
            )
            db_index = db_index+1

class GridEvalDataset(EmbeddableDataset):
    def __init__(self,
        hf_dataset_config: HuggingfaceDatasetConfig,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        palette_extractor: PaletteExtractor,
        histogram_extractor: HistogramExtractor,
        db_indexes: list[int],
        prompts: list[str],
        folder: Path
    ):
        super().__init__(hf_dataset_config, lda, text_encoder, palette_extractor, histogram_extractor, folder)
        self.db_indexes = db_indexes
        self.prompts = prompts
    
    def __len__(self):
        return len(self.db_indexes) * len(self.prompts)
    
    def __getitem__(self, index: int) -> BatchInput:
        db_index = self.db_indexes[index // len(self.prompts)]
        prompt = self.prompts[index % len(self.prompts)]
        hf_item = self.hf_dataset[db_index]
        return load_batch_from_hf(self.folder, hf_item)

class ColorDataset(EmbeddableDataset):
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, index: int) -> BatchInput:
        hf_item = self.hf_dataset[index]
        return load_batch_from_hf(self.folder, hf_item)
        
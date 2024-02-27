from ast import Call
from functools import cache
from math import e
from multiprocessing import process
from refiners.training_utils.trainer import Batch
from torch.utils.data import Dataset
from typing import Any, Dict, cast, TypedDict, Callable
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig, load_hf_dataset, HuggingfaceDataset
from tqdm import tqdm
from functools import cached_property
from ip_adapter_palette.types import BatchInput
from ip_adapter_palette.palette_adapter import PaletteExtractor
from ip_adapter_palette.histogram import HistogramExtractor
from torchvision import transforms
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
    db_index: int

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
            filename = dl_manager.download(url)
            return {"image": filename}

        hf_dataset = hf_dataset.map( # type: ignore
            function=download_image,
            input_columns=["photo_image_url"],
            fn_kwargs={
                "dl_manager": DownloadManager(),
            },
            batched=True,
            num_proc=os.cpu_count()
        )

        hf_dataset = hf_dataset.cast_column(
            column="image",
            feature=DatasetImage(),
        )
        hf_dataset = hf_dataset.add_column("db_index", range(len(hf_dataset)))

        return hf_dataset


def hf_item_to_batch_input(hf_item: HfItem, folder: Path) -> BatchInput:
    return load_batch_from_hf(folder, hf_item)

def filter_non_saved_latents_saved(batch_hf_items: list[HfItem], folder: Path) -> list[HfItem]:
    out : list[HfItem] = []
    for hf_item in batch_hf_items:
        filename = folder / f"{hf_item['photo_id']}.pt"
        if not filename.exists():
            out.append(hf_item)
    return out

def save_batch_as_latents(
        batch: BatchInput, 
        force: bool,
        folder: Path
    ) -> None:

    for batch_item in batch:
        filename = folder / f"{batch_item.id()}.pt"

        if filename.exists() and not force:
            logger.debug(f"Skipping {filename}. Already exists, change this behavior with --force.")
            continue
        
        value_dict = batch_item.to_dict()
        save(value_dict,filename)

def load_batch_from_hf(folder: Path, hf_item: HfItem) -> BatchInput:
    photo_id = hf_item['photo_id']
    filename = folder / f"{photo_id}.pt"
    return BatchInput.load_file(filename)


class EmbeddableDataset(Dataset[BatchInput]):
    def __init__(self,
        hf_dataset_config: HuggingfaceDatasetConfig,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        palette_extractor_weighted: PaletteExtractor,
        histogram_extractor: HistogramExtractor,
        folder: Path
    ):
        self.hf_dataset_config = hf_dataset_config
        self.hf_dataset = build_hf_dataset(hf_dataset_config)
        self.lda = lda
        self.text_encoder = text_encoder
        self.folder = folder
        self.palette_extractor_weighted = palette_extractor_weighted
        self.histogram_extractor = histogram_extractor
        self.process_image = self.build_image_processor(hf_dataset_config)
    
    def get_processed_hf_item(self, index: int) -> HfItem:
        base_hf_item = self.hf_dataset[index]
        image = self.process_image(base_hf_item['image'])

        return HfItem(
            photo_id = base_hf_item['photo_id'],
            image = image,
            caption = base_hf_item['caption'],
            db_index = base_hf_item['db_index']
        )
    def build_image_processor(self, config: HuggingfaceDatasetConfig) -> Callable[[Image.Image], Image.Image]:
        center_crop = transforms.CenterCrop(config.resize_image_max_size)

        def process_image(img: Image.Image) -> Image.Image:
            if img.mode != "RGB":
                img = img.convert("RGB")                
            img = resize_image(img, config.resize_image_min_size, config.resize_image_max_size)
            
            return center_crop(img)
        return process_image
    
    def build_batch_from_hf_items(
            self,
            hf_items: list[HfItem],
            lda: LatentDiffusionAutoencoder,
            text_encoder: CLIPTextEncoderL,
            palette_extractor: PaletteExtractor,
            histogram_extractor: HistogramExtractor,
            force: bool,
            config: HuggingfaceDatasetConfig
        ) -> BatchInput:

        images = [hf_item['image'] for hf_item in hf_items]
        processed_images = [self.process_image(image) for image in images]
        source_prompts = [hf_item['caption'] for hf_item in hf_items]
        return BatchInput(
            source_palettes_weighted = [palette_extractor(processed_image) for processed_image in processed_images],
            source_prompts = source_prompts,
            db_indexes = [hf_item['db_index'] for hf_item in hf_items],
            photo_ids = [hf_item['photo_id'] for hf_item in hf_items],
            source_text_embeddings = text_encoder(source_prompts),
            source_latents = lda.images_to_latents(processed_images),
            source_histograms = histogram_extractor.images_to_histograms(processed_images)
        )

    def precompute_embeddings(self, force: bool = False, batch_size: int = 1) -> None:
        num_workers = os.cpu_count()
        if num_workers is None:
            num_workers = 1
        
        def collate_fn(batch: list[HfItem]) -> list[HfItem]:
            return batch
        
        dataloader = DataLoader(
            cast(Dataset, self.hf_dataset), 
            batch_size=batch_size, 
            num_workers= num_workers,
            collate_fn=collate_fn
        )
        for hf_items in tqdm(dataloader): # type: ignore
            if not force:
                filtered = filter_non_saved_latents_saved(hf_items, self.folder)
            else:
                filtered = hf_items
            if len(filtered) == 0:
                continue
            batch = self.build_batch_from_hf_items(
                filtered,
                self.lda, 
                self.text_encoder, 
                self.palette_extractor_weighted,
                self.histogram_extractor,
                force, 
                self.hf_dataset_config
            )
            save_batch_as_latents(batch, force, self.folder)

class GridEvalDataset(EmbeddableDataset):
    def __init__(self,
        hf_dataset_config: HuggingfaceDatasetConfig,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        palette_extractor_weighted: PaletteExtractor,
        histogram_extractor: HistogramExtractor,
        db_indexes: list[int],
        prompts: list[str],
        folder: Path
    ):
        super().__init__(hf_dataset_config, lda, text_encoder, palette_extractor_weighted, histogram_extractor, folder)
        self.db_indexes = db_indexes
        self.prompts = prompts
    
    def __len__(self):
        return len(self.db_indexes) * len(self.prompts)
    
    @cached_property
    def prompts_embeddings(self) -> Tensor:
        return self.text_encoder(self.prompts)
    
    def __getitem__(self, index: int) -> BatchInput:
        db_index = self.db_indexes[index // len(self.prompts)]
        prompt_index = index % len(self.prompts)
        prompt = self.prompts[prompt_index]
        hf_item = self.hf_dataset[db_index]
        item = load_batch_from_hf(self.folder, hf_item)
        if len(item) != 1:
            raise ValueError(f"Loading failed, expected 1 item, got {len(item)}")
        item.source_prompts = [prompt]
        item.source_text_embeddings = self.prompts_embeddings[prompt_index:prompt_index+1]
        return item


class ColorDataset(EmbeddableDataset):
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, index: int) -> BatchInput:
        hf_item = self.hf_dataset[index]
        return load_batch_from_hf(self.folder, hf_item)

class ColorIndexesDataset(EmbeddableDataset):
    def __init__(self,
        hf_dataset_config: HuggingfaceDatasetConfig,
        lda: LatentDiffusionAutoencoder,
        text_encoder: CLIPTextEncoderL,
        palette_extractor_weighted: PaletteExtractor,
        histogram_extractor: HistogramExtractor,
        folder: Path,
        db_indexes: list[int]
    ):
        super().__init__(hf_dataset_config, lda, text_encoder, palette_extractor_weighted, histogram_extractor, folder)
        self.db_indexes = db_indexes
    
    def __len__(self):
        return len(self.db_indexes)
    
    def __getitem__(self, index: int) -> BatchInput:
        db_index = self.db_indexes[index]
        hf_item = self.hf_dataset[db_index]
        return load_batch_from_hf(self.folder, hf_item)
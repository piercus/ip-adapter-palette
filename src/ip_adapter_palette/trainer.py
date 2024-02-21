from functools import cached_property
from typing import cast, Any
from loguru import logger
import random
from ip_adapter_palette.callback import (
    SaveBestModel,
    SaveBestModelConfig,
)
from ip_adapter_palette.latent_diffusion import SD1TrainerMixin
from refiners.fluxion import load_from_safetensors
from refiners.fluxion.utils import no_grad
from ip_adapter_palette.palette_adapter import SD1PaletteAdapter, PaletteEncoder
from refiners.training_utils import (
    register_model,
    register_callback,
)
from ip_adapter_palette.config import Config, IPAdapterConfig

from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet

from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbMixin
import torch
from torch.nn import functional as F

from ip_adapter_palette.types import BatchInput
from ip_adapter_palette.config import PaletteEncoderConfig
from refiners.training_utils.huggingface_datasets import load_hf_dataset

from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, DownloadManager, Image


class SD1IPPalette(Trainer[Config, BatchInput], WandbMixin, SD1TrainerMixin):
    
    @register_model()
    def palette_encoder(self, config: PaletteEncoderConfig) -> PaletteEncoder:
        logger.info("Loading Palette Encoder.")
        
        # weights = load_from_safetensors(config.weights)
        
        palette_encoder = PaletteEncoder(
            feedforward_dim=config.feedforward_dim,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            embedding_dim=config.embedding_dim,
            max_colors=config.max_colors,
            mode=config.mode,
            weighted_palette=config.weighted_palette
        )

        return palette_encoder
    
    @register_model()
    def ip_adapter(self, config: IPAdapterConfig) -> SD1PaletteAdapter[SD1UNet]:
        logger.info("Loading IP Adapter.")
        if config.weights is not None:
            weights = load_from_safetensors(config.weights)
        else:
            weights = None
        
        ip_adapter = SD1PaletteAdapter(
            self.unet,
            palette_encoder = self.palette_encoder,
            weights=weights
        ).inject()
        
        for adapter in ip_adapter.sub_adapters:
            adapter.image_key_projection.requires_grad_(True)
            adapter.image_value_projection.requires_grad_(True)

        logger.info("IP Adapter loaded.")

        return ip_adapter


    @register_callback()
    def save_best_model(self, config: SaveBestModelConfig) -> SaveBestModel:
        return SaveBestModel(config)

    @cached_property
    def data(self) -> list[BatchInput]:
        return [
            BatchInput.load_file(batch).to(device=self.device, dtype=self.dtype)  # type: ignore
            for batch in self.config.data.rglob("*.pt")
        ]

    @cached_property
    @no_grad()
    def unconditional_text_embedding(self) -> torch.Tensor:
        self.text_encoder.to(device=self.device)
        embedding = self.text_encoder("")
        self.text_encoder.to(device="cpu")
        return embedding

    @cached_property
    @no_grad()
    def unconditional_palette(self) -> torch.Tensor:
        self.palette_encoder.to(device=self.device)
        embedding = self.palette_encoder([])
        self.palette_encoder.to(device="cpu")
        return embedding

    def get_item(self, index: int) -> BatchInput:
        
        item = self.data[index]
        if (
            random.random()
            < self.config.latent_diffusion.unconditional_sampling_probability
        ):
            item = BatchInput(
                source_palettes = self.unconditional_palette,
                source_prompts = self.unconditional_text_embedding,
                source_images = item['source_images'],
                source_latents = item['source_latents'],
                db_indexes = item['db_indexes'],
                text_embeddings = item['text_embeddings']
            )
        return item

    @classmethod
    def load_file(cls, file_path: str) -> "BatchInput":
        return BatchInput.load_file(file_path)

    def collate_fn(self, batch: list[BatchInput]) -> BatchInput:
        return BatchInput.collate_fn(batch)

    @property
    def dataset_length(self) -> int:
        return len(self.hf_dataset)

    def compute_loss(self, batch: BatchInput) -> torch.Tensor:
        source_latents, text_embeddings, source_palettes = (
            batch['source_latents'],
            batch['text_embeddings'],
            batch['source_palettes']
        )
        if type(text_embeddings) is not torch.Tensor:
            raise ValueError(f"Text embeddings should be a tensor, not {type(text_embeddings)}")
        
        if type(source_latents) is not torch.Tensor:
            raise ValueError(f"Latents should be a tensor, not {type(source_latents)}")

        timestep = self.sample_timestep(source_latents.shape[0])
        noise = self.sample_noise(source_latents.shape)
        noisy_latents = self.add_noise_to_latents(source_latents, noise)
        palette_embeddings = self.palette_encoder(source_palettes)
        self.unet.set_timestep(timestep)
        self.unet.set_clip_text_embedding(text_embeddings)
        self.ip_adapter.set_palette_embedding(palette_embeddings)
        prediction = self.unet(noisy_latents)
        loss = F.mse_loss(input=prediction, target=noise)
        return loss
    
    @cached_property
    def hf_dataset(self) -> Dataset:
        hf_dataset = load_hf_dataset(
            self.config.dataset.hf_repo,
            self.config.dataset.revision,
            self.config.dataset.split,
            self.config.dataset.use_verification
        )
        def download_image(url: str | list[str], dl_manager: DownloadManager):
            img = dl_manager.download(url)
            return {"image": img}

        hf_dataset = hf_dataset.map(
            function=download_image,
            input_columns=["photo_image_url"],
            fn_kwargs={
                "dl_manager": DownloadManager(),
            },
            batched=True,
            num_proc=4,
        )
        hf_dataset = hf_dataset.cast_column(
            column="image",
            feature=Image(),
        )
        return cast(Dataset, hf_dataset)
    
    def precompute(self) -> None:
        def collate_fn(batch: list) -> Any:
           
            return {
                "photo_id": [item["photo_id"] for item in batch],
                "image": [item["image"] for item in batch],
                "caption": [item["caption"] for item in batch]
            }
        
        dataloader = DataLoader(
            dataset=self.hf_dataset, 
            batch_size=self.config.training.batch_size, 
            collate_fn=collate_fn
        )
        
        for batch in dataloader:
            self.precompute_batch(batch)
    
    def precompute_batch(self, batch: list) -> None:
        folder = self.config.data
        for (photo_id, image, caption) in zip(batch['photo_id'], batch['image'], batch['caption']):
            
            latents = self.lda.image_to_latents(image)
            text_embedding = self.text_encoder(caption)
            filename = folder / f"{photo_id}.pt"
            torch.save(
                {
                    "latents": latents,
                    "text_embedding": text_embedding
                },
                filename
            )
        
    def compute_evaluation(self) -> None:
        raise NotImplementedError("Evaluation not implemented.")

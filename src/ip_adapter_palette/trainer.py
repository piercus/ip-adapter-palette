from functools import cached_property
from typing import Any, Callable
from unittest import result
from loguru import logger
import random
import numpy as np
from requests import get
from ip_adapter_palette import histogram_auto_encoder
from ip_adapter_palette.callback import (
    MonitorGradient,
    MonitorGradientConfig,
    OffloadToCPU,
    OffloadToCPUConfig,
    SaveBestModel,
    SaveBestModelConfig,
    LogModelParam,
    LogModelParamConfig,
    MonitorTime,
    MonitorTimeConfig,
    TimestepLossRescaler,
    TimestepLossRescalerConfig,
)
from ip_adapter_palette.evaluation.grid_evaluation import GridEvaluationCallback, GridEvaluationConfig
from ip_adapter_palette.evaluation.mmd_evaluation import MmdEvaluationCallback
from ip_adapter_palette.evaluation.visual_evaluation import VisualEvaluationCallback, VisualEvaluationConfig
from ip_adapter_palette.histogram_auto_encoder import HistogramAutoEncoder
from ip_adapter_palette.latent_diffusion import SD1TrainerMixin
from refiners.fluxion import load_from_safetensors
from refiners.fluxion.utils import no_grad
from ip_adapter_palette.metrics.mmd import mmd
from ip_adapter_palette.palette_adapter import SD1PaletteAdapter, PaletteEncoder, PaletteExtractor, Palette, Color
from ip_adapter_palette.histogram import HistogramDistance, HistogramExtractor, histogram_to_histo_channels
from ip_adapter_palette.metrics.palette import batch_image_palette_metrics, ImageAndPalette

from refiners.training_utils import (
    register_model,
    register_callback,
)
import os
from ip_adapter_palette.config import Config, HistogramAutoEncoderConfig, IPAdapterConfig, MmdEvaluationConfig
from ip_adapter_palette.datasets import ColorDataset, GridEvalDataset
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet

from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbMixin, WandbLoggable
import torch
from torch.nn import functional as F, Module as TorchModule
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale # type: ignore
from torch import Tensor, tensor, randn, cat
from ip_adapter_palette.types import BatchInput
from ip_adapter_palette.config import PaletteEncoderConfig
from refiners.training_utils.huggingface_datasets import load_hf_dataset, HuggingfaceDatasetConfig

from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, DownloadManager, Image as DatasetImage # type: ignore
from loguru import logger
from PIL import Image, ImageDraw
from tqdm import tqdm
from refiners.training_utils.common import scoped_seed
from refiners.fluxion.utils import tensor_to_images, tensor_to_image, images_to_tensor
from typing import TypedDict, Tuple
from torch.nn.functional import mse_loss

class PaletteTrainer(Trainer[Config, BatchInput], WandbMixin, SD1TrainerMixin):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
    
    @register_model()
    def histogram_auto_encoder(self, config: HistogramAutoEncoderConfig) -> HistogramAutoEncoder:
        logger.info("Loading Histogram Auto Encoder.")
        
        histogram_auto_encoder = HistogramAutoEncoder(
            latent_dim=config.latent_dim,
            resnet_sizes=config.resnet_sizes,
            n_down_samples=config.n_down_samples,
            color_bits=config.color_bits,
            num_groups=config.num_groups,
            device=self.device,
            dtype=self.dtype
        )

        histogram_auto_encoder.requires_grad_(False)

        return histogram_auto_encoder
    
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
        )
        if config.weights is None:
            ip_adapter.zero_init()

        ip_adapter.inject()
        
        for adapter in ip_adapter.sub_adapters:
            adapter.image_key_projection.requires_grad_(True)
            adapter.image_value_projection.requires_grad_(True)

        logger.info("IP Adapter loaded.")

        return ip_adapter


    @register_callback()
    def save_best_model(self, config: SaveBestModelConfig) -> SaveBestModel:
        return SaveBestModel(config)
    
    @register_callback()
    def log_model_params(self, config: LogModelParamConfig) -> LogModelParam:
        return LogModelParam()
    
    @register_callback()
    def monitor_time(self, config: MonitorTimeConfig) -> MonitorTime:
        return MonitorTime(config)
    
    @register_callback()
    def monitor_gradient(self, config: MonitorGradientConfig) -> MonitorGradient:
        return MonitorGradient(config)
    
    @register_callback()
    def offload_to_cpu(self, config: OffloadToCPUConfig) -> OffloadToCPU:
        return OffloadToCPU(config)
    
    @register_callback()
    def timestep_loss_rescaler(self, config: TimestepLossRescalerConfig) -> TimestepLossRescaler:
        return TimestepLossRescaler(config)
    
    @register_callback()
    def grid_evaluation(self, config: GridEvaluationConfig) -> GridEvaluationCallback:
        return GridEvaluationCallback(config)
    
    @register_callback()
    def mmd_evaluation(self, config: MmdEvaluationConfig) -> MmdEvaluationCallback:
        return MmdEvaluationCallback(config)

    @register_callback()
    def visual_evaluation(self, config: VisualEvaluationConfig) -> VisualEvaluationCallback:
        return VisualEvaluationCallback(config)
    
    @cached_property
    def data(self) -> list[BatchInput]:
        return [
            BatchInput.load_file(batch).to(device=self.device, dtype=self.dtype)  # type: ignore
            for batch in self.config.data.rglob("*.pt")
        ]
    
    def collate_fn(self, batch: list[BatchInput]) -> BatchInput:
        return BatchInput.collate(batch)

    @cached_property
    @no_grad()
    def unconditional_text_embedding(self) -> torch.Tensor:
        self.text_encoder.to(device=self.device)
        embedding = self.text_encoder("")
        self.text_encoder.to(device="cpu")
        return embedding

    @cached_property
    @no_grad()
    def unconditional_palette(self) -> list[Palette]:
        return [[]]

    def get_item(self, index: int) -> BatchInput:
        
        item = self.data[index]
        if (
            random.random()
            < self.config.latent_diffusion.unconditional_sampling_probability
        ):
            item = BatchInput(
                source_palettes_weighted = self.unconditional_palette * len(item),
                source_prompts = [""]*len(item),
                source_latents = item.source_latents,
                db_indexes = item.db_indexes,
                photo_ids = item.photo_ids,
                source_text_embeddings = self.unconditional_text_embedding.repeat(len(item), 1, 1),
                source_histograms = item.source_histograms
            )
        return item

    def collate(self, batch: list[BatchInput]) -> BatchInput:
        return BatchInput.collate(batch)

    @property
    def dataset_length(self) -> int:
        return len(self.hf_train_dataset) # type: ignore
    
    def process_palettes(self, palettes: list[Palette]) -> list[Palette]:
        if self.config.palette_encoder.weighted_palette:
            return [
                [(palette_cluster[0], 1.0/len(palette))  for palette_cluster in palette]
                for palette in palettes 
            ]
        return palettes
    
    def compute_loss(self, batch: BatchInput) -> torch.Tensor:
        source_latents, text_embeddings, source_palettes, source_histograms = (
            batch.source_latents,
            batch.source_text_embeddings,
            batch.source_palettes_weighted,
            batch.source_histograms
        )
        if type(text_embeddings) is not torch.Tensor:
            raise ValueError(f"Text embeddings should be a tensor, not {type(text_embeddings)}")
        
        if type(source_latents) is not torch.Tensor:
            raise ValueError(f"Latents should be a tensor, not {type(source_latents)}")

        timestep = self.sample_timestep(source_latents.shape[0])
        self.timestep = timestep
        noise = self.sample_noise(source_latents.shape)
        noisy_latents = self.add_noise_to_latents(source_latents, noise)
        
        self.unet.set_timestep(timestep)

        match self.config.mode:
            case "palette":
                processed = self.process_palettes(source_palettes)
                palette_embeddings = self.palette_encoder(processed)
                self.ip_adapter.set_palette_embedding(palette_embeddings)
                self.unet.set_clip_text_embedding(text_embeddings)
            case "text_embedding":
                self.ip_adapter.set_palette_embedding(text_embeddings)
                self.unet.set_clip_text_embedding(self.unconditional_text_embedding)
            case "histogram":
                histogram_embeddings = self.histogram_auto_encoder.encode_sequence(source_histograms)
                self.ip_adapter.set_palette_embedding(histogram_embeddings)
                self.unet.set_clip_text_embedding(text_embeddings)
        
        prediction = self.unet(noisy_latents)
        loss = F.mse_loss(input=prediction, target=noise, reduction='none')
        return loss
    
    @cached_property
    def hf_train_dataset(self) -> ColorDataset:
        return ColorDataset(
            hf_dataset_config=self.config.dataset,
            lda=self.lda,
            text_encoder=self.text_encoder,
            palette_extractor_weighted=self.palette_extractor_weighted,
            histogram_extractor=self.histogram_extractor,
            folder=self.config.data
        )
    
    def precompute(self, batch_size: int=1, force: bool=False) -> None:
        self._call_callbacks(event_name="on_precompute_start")
        self.hf_train_dataset.precompute_embeddings(force=force, batch_size=batch_size)
        self._call_callbacks(event_name="on_precompute_end")


    def compute_evaluation(
        self
    ) -> None:
        
        pass

    @cached_property
    def palette_extractor_weighted(self) -> PaletteExtractor:
        return PaletteExtractor(
            size=self.config.palette_encoder.max_colors,
            weighted_palette=True
        )

    @cached_property
    def palette_extractor_unweighted(self) -> PaletteExtractor:
        return PaletteExtractor(
            size=self.config.palette_encoder.max_colors,
            weighted_palette=False
        )
    @cached_property
    def histogram_distance(self) -> HistogramDistance:
        return HistogramDistance(color_bits=self.config.grid_evaluation.color_bits)
    
    @cached_property
    def histogram_extractor(self) -> HistogramExtractor:
        return HistogramExtractor(color_bits=self.config.grid_evaluation.color_bits)


    def set_adapter_values(self, batch: BatchInput) -> None:

        clip_text_embedding = cat(tensors=(batch.source_text_embeddings, batch.source_text_embeddings))
        match self.config.mode:
            case "palette":
                self.ip_adapter.set_palette_embedding(
                    self.palette_encoder.compute_palette_embedding(
                        self.process_palettes(batch.source_palettes_weighted)
                    )
                )
            case "text_embedding":
                self.ip_adapter.set_palette_embedding(
                    clip_text_embedding
                )
            case "histogram":
                self.ip_adapter.set_palette_embedding(
                    self.histogram_auto_encoder.compute_histogram_embedding(
                        batch.source_histograms
                    )
                )
    
    @scoped_seed(5)
    def batch_inference(self, batch: BatchInput, same_seed: bool = True) -> Tensor:
        batch_size = len(batch.source_prompts)
        
        logger.info(f"Inference on {batch_size} images for {batch_size}")
        
        if same_seed:
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
            x = x.repeat(batch_size, 1, 1, 1)
        else: 
            x = randn(batch_size, 4, 64, 64, dtype=self.dtype, device=self.device)

        self.set_adapter_values(batch)
        uncond = self.unconditional_text_embedding.repeat(batch_size,1,1)
        clip_text_embedding = cat(tensors=(uncond, uncond))
        
        for step in self.sd.steps:
            x = self.sd(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                condition_scale = self.config.grid_evaluation.condition_scale
            )

        return x
    
    

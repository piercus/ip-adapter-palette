from functools import cached_property
from typing import Any, Callable
from unittest import result
from loguru import logger
import random
import numpy as np
from requests import get
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
from ip_adapter_palette.config import Config, IPAdapterConfig, MmdEvaluationConfig
from ip_adapter_palette.datasets import ColorDataset, GridEvalDataset
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet

from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbMixin, WandbLoggable
import torch
from torch.nn import functional as F, Module as TorchModule
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale # type: ignore
from torch import Tensor, tensor, randn, cat
from ip_adapter_palette.types import BatchInput, BatchOutput
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
    
    # @register_callback()
    # def mmd_evaluation(self, config: MmdEvaluationConfig) -> MmdEvaluation:
    #     return MmdEvaluation(config)

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
                source_palettes = self.unconditional_palette * len(item),
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

    def compute_loss(self, batch: BatchInput) -> torch.Tensor:
        source_latents, text_embeddings, source_palettes = (
            batch.source_latents,
            batch.source_text_embeddings,
            batch.source_palettes
        )
        if type(text_embeddings) is not torch.Tensor:
            raise ValueError(f"Text embeddings should be a tensor, not {type(text_embeddings)}")
        
        if type(source_latents) is not torch.Tensor:
            raise ValueError(f"Latents should be a tensor, not {type(source_latents)}")

        timestep = self.sample_timestep(source_latents.shape[0])
        self.timestep = timestep
        noise = self.sample_noise(source_latents.shape)
        noisy_latents = self.add_noise_to_latents(source_latents, noise)
        palette_embeddings = self.palette_encoder(source_palettes)
        self.unet.set_timestep(timestep)
        self.unet.set_clip_text_embedding(text_embeddings)
        self.ip_adapter.set_palette_embedding(palette_embeddings)
        prediction = self.unet(noisy_latents)
        loss = F.mse_loss(input=prediction, target=noise, reduction='none')
        return loss
    
    @cached_property
    def hf_train_dataset(self) -> ColorDataset:
        return ColorDataset(
            hf_dataset_config=self.config.dataset,
            lda=self.lda,
            text_encoder=self.text_encoder,
            palette_extractor=self.palette_extractor,
            histogram_extractor=self.histogram_extractor,
            folder=self.config.data
        )
    
    def precompute(self, batch_size: int=1, force: bool=False) -> None:
        self.hf_train_dataset.precompute_embeddings(force=force, batch_size=batch_size)
        self.grid_eval_dataset.precompute_embeddings(force=force, batch_size=batch_size)
    
    def compute_grid_evaluation(
        self
    ) -> None:
        
        per_prompts : dict[str, BatchOutput] = {}
        images : dict[str, WandbLoggable] = {}
                
        for batch in self.grid_eval_dataloader:
            results = self.batch_inference(batch.to(device=self.device, dtype=self.dtype))
        
            for prompt in list(set(results.source_prompts)):
                batch = results.get_prompt(prompt)
                if prompt not in per_prompts:
                    per_prompts[prompt] = batch
                else:
                    per_prompts[prompt] = BatchOutput.collate([
                        per_prompts[prompt],
                        batch
                    ])
        
        for prompt in per_prompts:
            self.wandb_log(data={
                f"inter_prompt_distance/{prompt}": self.image_distances(per_prompts[prompt])
            })
            image = self.draw_palette_cover_image(per_prompts[prompt])
            image_name = f"eval_images/{prompt}"
            images[image_name] = image
            
        all_results = BatchOutput.collate(list(per_prompts.values()))
        
        # images[f"eval_images/all"] = self.draw_cover_image(all_results)
        self.wandb_log(data=images)

        self.batch_metrics(all_results, prefix="eval")
    
    def compute_evaluation(
        self
    ) -> None:
        
        self.compute_grid_evaluation()
        # self.compute_image_quality_evaluation()

    def image_distances(self, batch: BatchOutput) -> float:
        images = images_to_tensor(batch.result_images)
        dist = tensor(0)
        for i in range(images.shape[0]):
            for j in range(i+1, images.shape[0]):
                dist = dist + mse_loss(images[i], images[j])
        
        return dist.item()
    @cached_property
    def palette_extractor(self) -> PaletteExtractor:
        return PaletteExtractor(
            size=self.config.palette_encoder.max_colors,
            weighted_palette=self.config.palette_encoder.weighted_palette
        )
    
    def draw_palette(self, palette: Palette, width: int, height: int) -> Image.Image:
        palette_img = Image.new(mode="RGB", size=(width, height))
        
        # sort the palette by weight
        current_x = 0
        for (color, weight) in palette:
            box_width = int(weight*width)            
            color_box = Image.fromarray(np.full((height, box_width, 3), color, dtype=np.uint8)) # type: ignore
            palette_img.paste(color_box, box=(current_x, 0))
            current_x+=box_width
            
        return palette_img
    
    def batch_metrics(self, results: BatchOutput, prefix: str = "palette-img") -> None:
        palettes : list[list[Color]] = []
        for p in results.source_palettes:
            palettes.append([cluster[0] for cluster in p])
        
        images = results.result_images
        
        batch_image_palette_metrics(
            self.wandb_log, 
            [
                ImageAndPalette({"image": image, "palette": palette})
                for image, palette in zip(images, palettes)
            ], 
            prefix
        )

        result_histo = results.result_histograms
        
        histo_metrics = self.histogram_distance.metrics(result_histo, results.source_histograms.to(result_histo.device))
        
        log_dict : dict[str, WandbLoggable] = {}
        for (key, value) in histo_metrics.items():
            log_dict[f"eval_histo/{key}"] = value.item()
        
        self.wandb_log(log_dict)  

    @cached_property
    def histogram_distance(self) -> HistogramDistance:
        return HistogramDistance(color_bits=self.config.grid_evaluation.color_bits)
    
    @cached_property
    def histogram_extractor(self) -> HistogramExtractor:
        return HistogramExtractor(color_bits=self.config.grid_evaluation.color_bits)
    
    @cached_property
    def grid_eval_dataset(self) -> GridEvalDataset:
        return GridEvalDataset(
            hf_dataset_config=self.config.eval_dataset,
            lda=self.lda,
            text_encoder=self.text_encoder,
            palette_extractor=self.palette_extractor,
            histogram_extractor=self.histogram_extractor,
            folder=self.config.data,
            db_indexes=self.config.grid_evaluation.db_indexes,
            prompts=self.config.grid_evaluation.prompts
        )
    @cached_property
    def grid_eval_dataloader(self) -> DataLoader[BatchInput]:
        # num_workers = os.cpu_count()
        # if num_workers is None:
        #     num_workers = 1
        
        logger.debug(f"Evaluation batch size is {self.config.grid_evaluation.batch_size}")
        
        return DataLoader(
            dataset=self.grid_eval_dataset, 
            batch_size=self.config.grid_evaluation.batch_size, 
            shuffle=False,
            collate_fn=BatchInput.collate, 
            #num_workers=num_workers
        )

    def eval_set_adapter_values(self, batch: BatchInput) -> None:
        self.ip_adapter.set_palette_embedding(
            self.palette_encoder.compute_palette_embedding(
                batch.source_palettes
            )
        )
    
    @scoped_seed(5)
    def batch_inference(self, batch: BatchInput, same_seed: bool = True) -> BatchOutput:
        batch_size = len(batch.source_prompts)
        
        logger.info(f"Inference on {batch_size} images for prompts/db_indexes: {batch.source_prompts}/{batch.db_indexes}")
        
        if same_seed:
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
            x = x.repeat(batch_size, 1, 1, 1)
        else: 
            x = randn(batch_size, 4, 64, 64, dtype=self.dtype, device=self.device)

        self.eval_set_adapter_values(batch)
        
        clip_text_embedding = cat(tensors=(self.unconditional_text_embedding.repeat(batch_size,1,1), batch.source_text_embeddings))
        
        for step in self.sd.steps:
            x = self.sd(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                condition_scale = self.config.grid_evaluation.condition_scale
            )

        # images = (self.lda.decode(x) + 1 )/2
        return self.build_results(batch, x)
    
    def get_grid_eval_images(self, db_indexes: list[int], photo_ids: list[str]) -> list[Image.Image]:
        images = []
        for (db_index, photo_id) in zip(db_indexes, photo_ids):
            hf_item = self.grid_eval_dataset.get_processed_hf_item(db_index)
            if hf_item['photo_id'] != photo_id:
                raise ValueError(f"Photo id mismatch: {hf_item['photo_id']} != {photo_id}")
            images.append(hf_item['image'])
        return images

    def build_results(self, batch: BatchInput, result_latents: Tensor) -> BatchOutput:
        result_images = self.lda.latents_to_images(result_latents)
        return BatchOutput(
            source_prompts=batch.source_prompts,
            db_indexes=batch.db_indexes,
            source_histograms=batch.source_histograms,
            source_palettes=batch.source_palettes,
            source_images=self.get_grid_eval_images(batch.db_indexes, batch.photo_ids),
            photo_ids = batch.photo_ids,
            source_text_embeddings= batch.source_text_embeddings,
            source_latents=batch.source_latents,
            result_histograms = self.histogram_extractor.images_to_histograms(result_images),
            result_latents = result_latents,
            result_images=result_images,         
            result_palettes=[self.palette_extractor(image, size=len(batch.source_palettes[i])) for i, image in enumerate(result_images)],
        )
    
    
    def draw_palette_cover_image(self, batch: BatchOutput) -> Image.Image:
        res_images = images_to_tensor(batch.result_images)
        (batch_size, _, height, width) = res_images.shape
        
        palette_img_size = width // self.config.palette_encoder.max_colors
        source_images = batch.source_images

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(2*width, (height+palette_img_size) * batch_size)
        )
        
        images = tensor_to_images(res_images)
        for i, image in enumerate(images):
            join_canvas_image.paste(source_images[i], box=(0, i*(height+palette_img_size)))
            join_canvas_image.paste(image, box=(width, i*(height+palette_img_size)))
            palette_out = batch.result_palettes[i]
            palette_int = batch.source_palettes[i]
            palette_out_img = self.draw_palette(palette_out, width, palette_img_size)
            palette_in_img = self.draw_palette(palette_int, width, palette_img_size)
            
            join_canvas_image.paste(palette_in_img, box=(0, i*(height+palette_img_size) + height))
            join_canvas_image.paste(palette_out_img, box=(width, i*(height+palette_img_size) + height))
        return join_canvas_image

    def draw_curves(self, res_histo: list[float], src_histo: list[float], color: str, width: int, height: int) -> Image.Image:
        histo_img = Image.new(mode="RGB", size=(width, height))
        
        draw = ImageDraw.Draw(histo_img)
        
        if len(res_histo) != len(src_histo):
            raise ValueError("The histograms must have the same length.")
        
        ratio = width/len(res_histo)
        semi_height = height//2
        
        scale_ratio = 5
                
        draw.line([
            (i*ratio, (1-res_histo[i]*scale_ratio)*semi_height + semi_height) for i in range(len(res_histo))
        ], fill=color, width=4)
        
        draw.line([
            (i*ratio, (1-src_histo[i]*scale_ratio)*semi_height) for i in range(len(src_histo))
        ], fill=color, width=1)
        
        return histo_img
    
    def draw_histogram_cover_image(self, batch: BatchOutput) -> Image.Image:
        res_images = images_to_tensor(batch.result_images)
        (batch_size, channels, height, width) = res_images.shape

        vertical_image = res_images.permute(0,2,3,1).reshape(1, height*batch_size, width, channels).permute(0,3,1,2)
        
        results_histograms = batch.result_histograms
        source_histograms = batch.source_histograms
        source_images = batch.source_images
        src_palettes = batch.source_palettes

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(width + width//2, height * batch_size)
        )
        res_image = tensor_to_image(vertical_image)
        
        join_canvas_image.paste(res_image, box=(width//2, 0))
        
        res_histo_channels = histogram_to_histo_channels(results_histograms)
        src_histo_channels = histogram_to_histo_channels(source_histograms)
        
        colors = ["red", "green", "blue"]
        
        for i in range(batch_size):
            image = source_images[i]
            join_canvas_image.paste(image.resize((width//2, height//2)), box=(0, height *i))
            
            source_image_palette = self.draw_palette(
                self.palette_extractor.from_histogram(source_histograms[i], color_bits= self.config.histogram_auto_encoder.color_bits, size=len(src_palettes[i])),
                width//2,
                height//16
            )
            join_canvas_image.paste(source_image_palette, box=(0, height *i + height//2))
            
            res_image_palette = self.draw_palette(
                self.palette_extractor.from_histogram(results_histograms[i], color_bits= self.config.histogram_auto_encoder.color_bits, size=len(src_palettes[i])),
                width//2,
                height//16
            )
            
            join_canvas_image.paste(res_image_palette, box=(0, height *i + (15*height)//16))

            for (color_id, color_name) in enumerate(colors):
                image_curve = self.draw_curves(
                    res_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    src_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    color_name,
                    width//2,
                    height//8
                )
                join_canvas_image.paste(image_curve, box=(0, height *i + height//2 + ((1+2*color_id)*height)//16))
                
        return join_canvas_image
    
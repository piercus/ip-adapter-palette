from functools import cached_property
from typing import cast, Any, Callable
from loguru import logger
import random
from ip_adapter_palette.callback import (
    SaveBestModel,
    SaveBestModelConfig,
)
from ip_adapter_palette.latent_diffusion import SD1TrainerMixin
from refiners.fluxion import load_from_safetensors
from refiners.fluxion.utils import no_grad
from ip_adapter_palette.palette_adapter import SD1PaletteAdapter, PaletteEncoder, PaletteExtractor, Palette, Color
from ip_adapter_palette.histogram import HistogramDistance, histogram_to_histo_channels
from ip_adapter_palette.metrics.palette import batch_image_palette_metrics, ImageAndPalette

from refiners.training_utils import (
    register_model,
    register_callback,
)
import os
from ip_adapter_palette.config import Config, IPAdapterConfig
from ip_adapter_palette.datasets import ColorDataset, GridEvalDataset
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet

from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbMixin, WandbLoggable
import torch
from torch.nn import functional as F, Module as TorchModule
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale # type: ignore
from torch import Tensor, tensor, randn
from ip_adapter_palette.types import BatchInput, BatchOutput
from ip_adapter_palette.config import PaletteEncoderConfig
from refiners.training_utils.huggingface_datasets import load_hf_dataset, HuggingfaceDatasetConfig

from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, DownloadManager, Image as DatasetImage # type: ignore
from loguru import logger
from PIL import Image, ImageDraw
from tqdm import tqdm
from refiners.training_utils.common import scoped_seed
from refiners.fluxion.utils import tensor_to_images, tensor_to_image
from typing import TypedDict, Tuple
from torch.nn.functional import mse_loss

class SD1IPPalette(Trainer[Config, BatchInput], WandbMixin, SD1TrainerMixin):
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
        return len(self.hf_dataset) # type: ignore

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
    def hf_train_dataset(self) -> ColorDataset:
        return ColorDataset(
            hf_dataset_config=self.config.train_dataset,
            lda=self.lda,
            text_encoder=self.text_encoder,
            folder=self.config.data
        )
    
    def precompute(self, batch_size: int=1, force: bool=False) -> None:
        self.hf_train_dataset.precompute_embeddings(force=force)    

    def compute_evaluation(
        self
    ) -> None:
        
        per_prompts : dict[str, BatchOutput] = {}
        images : dict[str, WandbLoggable] = {}
        
        all_results : BatchOutput = BatchOutput.empty()
        
        for batch in self.eval_dataloader:
            results = self.compute_batch_evaluation(batch)
        
            for prompt in list(set(results.source_prompts)):
                batch = results.get_prompt(prompt)
                if prompt not in per_prompts:
                    per_prompts[prompt] = batch
                else:
                    per_prompts[prompt] = BatchOutput.collate_fn([
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
            
        all_results = BatchOutput.collate_fn(list(per_prompts.values()))
        
        # images[f"eval_images/all"] = self.draw_cover_image(all_results)
        self.wandb_log(data=images)

        self.batch_metrics(all_results, prefix="eval")

    def image_distances(self, batch: BatchOutput) -> float:
        images = batch["result_images"]
        if type(images) is not torch.Tensor:
            raise ValueError(f"Images should be a tensor, not {type(images)}")
        dist = tensor(0)
        for i in range(images.shape[0]):
            for j in range(i+1, images.shape[0]):
                dist = dist + mse_loss(images[i], images[j])
        
        return dist.item()
    @cached_property
    def color_palette_extractor(self) -> PaletteExtractor:
        return PaletteExtractor(
            size=self.config.color_palette.max_colors,
            weighted_palette=self.config.color_palette.weighted_palette
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
        for p in results["source_palettes"]:
            p = cast(Palette, p)
            palettes.append([cluster[0] for cluster in p])
        
        images = tensor_to_images(cast(Tensor, results["result_images"]))
        
        batch_image_palette_metrics(
            self.wandb_log, 
            [
                ImageAndPalette({"image": image, "palette": palette})
                for image, palette in zip(images, palettes)
            ], 
            prefix
        )

        result_histo = cast(Tensor, results["result_histograms"])
        
        histo_metrics = self.histogram_distance.metrics(result_histo, cast(Tensor, results["source_histograms"]).to(result_histo.device))
        
        log_dict : dict[str, WandbLoggable] = {}
        for (key, value) in histo_metrics.items():
            log_dict[f"eval_histo/{key}"] = value.item()
        
        self.wandb_log(log_dict)  

    @cached_property
    def histogram_distance(self) -> HistogramDistance:
        return HistogramDistance(color_bits=self.config.evaluation.color_bits)
    
    @cached_property
    def grid_eval_dataset(self) -> GridEvalDataset:
        return GridEvalDataset(
            hf_dataset_config=self.config.eval_dataset,
            lda=self.lda,
            text_encoder=self.text_encoder,
            folder=self.config.data,
            db_indexes=self.config.evaluation.db_indexes,
            prompts=self.config.evaluation.prompts
        )
    @cached_property
    def eval_dataloader(self) -> DataLoader[BatchInput]:
             
        return DataLoader(
            dataset=self.grid_eval_dataset, 
            batch_size=self.config.evaluation.batch_size, 
            shuffle=False,
            collate_fn=BatchInput.collate_fn, 
            num_workers=self.config.training.num_workers
        )

    def eval_set_adapter_values(self, batch: BatchInput) -> None:
        self.ip_adapter.set_palette_embedding(
            self.palette_encoder.compute_palette_embedding(
                cast(list[Palette], batch['source_palettes'])
            )
        )
    
    @scoped_seed(5)
    def compute_batch_evaluation(self, batch: BatchInput, same_seed: bool = True) -> BatchOutput:
        batch_size = len(batch["source_prompts"])
        
        logger.info(f"Generating {batch_size} images for prompts/db_indexes: {batch['source_prompts']}/{batch['db_indexes']}")
        
        if same_seed:
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
            x = x.repeat(batch_size, 1, 1, 1)
        else: 
            x = randn(batch_size, 4, 64, 64, dtype=self.dtype, device=self.device)

        self.eval_set_adapter_values(batch)
        
        clip_text_embedding = batch["text_embeddings"]
        
        for step in self.sd.steps:
            x = self.sd(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                condition_scale = self.config.evaluation.condition_scale
            )

        images = (self.lda.decode(x) + 1 )/2
        return self.build_results(batch, images)
    
    def build_results(self, batch: BatchInput, result_images: Tensor) -> BatchOutput:
        ...
    
    
    def draw_palette_cover_image(self, batch: BatchOutput) -> Image.Image:
        res_images = cast(Tensor, batch["result_images"])
        (batch_size, _, height, width) = res_images.shape
        
        palette_img_size = width // self.config.color_palette.max_colors
        source_images = cast(list[Image.Image], batch['source_images'])

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(2*width, (height+palette_img_size) * batch_size)
        )
        
        images = tensor_to_images(res_images)
        for i, image in enumerate(images):
            join_canvas_image.paste(source_images[i], box=(0, i*(height+palette_img_size)))
            join_canvas_image.paste(image, box=(width, i*(height+palette_img_size)))
            palette_out = cast(list[Palette], batch["result_palettes"])[i]
            palette_int = cast(list[Palette], batch["source_palettes"])[i]
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
        res_images = cast(Tensor, batch["result_images"])
        (batch_size, channels, height, width) = res_images.shape

        vertical_image = res_images.permute(0,2,3,1).reshape(1, height*batch_size, width, channels).permute(0,3,1,2)
        
        results_histograms = cast(Tensor, batch['result_histograms'])
        source_histograms = cast(Tensor, batch['source_histograms'])
        source_images = cast(list[Image.Image], batch['source_images'])
        src_palettes = cast(list[Palette], batch['source_palettes'])

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(width + width//2, height * batch_size)
        )
        res_image = tensor_to_image(vertical_image)
        
        join_canvas_image.paste(res_image, box=(width//2, 0))
        
        res_histo_channels = histogram_to_histo_channels(results_histograms)
        src_histo_channels = histogram_to_histo_channels(source_histograms)
        
        colors = ["red", "green", "blue"]
        
        for i in range(batch_size):
            join_canvas_image.paste(source_images[i].resize((width//2, height//2)), box=(0, height *i))
            
            source_image_palette = self.draw_palette(
                self.color_palette_extractor.from_histogram(source_histograms[i], color_bits= self.config.histogram_auto_encoder.color_bits, size=len(src_palettes[i])),
                width//2,
                height//16
            )
            join_canvas_image.paste(source_image_palette, box=(0, height *i + height//2))
            
            res_image_palette = self.draw_palette(
                self.color_palette_extractor.from_histogram(results_histograms[i], color_bits= self.config.histogram_auto_encoder.color_bits, size=len(src_palettes[i])),
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
    
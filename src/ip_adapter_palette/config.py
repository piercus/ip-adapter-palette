from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.training_utils.wandb import WandbConfig
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from torch import embedding
from ip_adapter_palette.callback import LogModelParamConfig, MonitorTimeConfig, MonitorGradientConfig, OffloadToCPUConfig, SaveBestModelConfig, TimestepLossRescalerConfig
from ip_adapter_palette.evaluation.fid_evaluation import FidEvaluationConfig
from ip_adapter_palette.evaluation.grid_evaluation import GridEvaluationConfig
from ip_adapter_palette.evaluation.mmd_evaluation import MmdEvaluationConfig
from ip_adapter_palette.evaluation.visual_evaluation import VisualEvaluationConfig

class LatentDiffusionConfig(BaseModel):
    unconditional_sampling_probability: float = 0.2
    offset_noise: float = 0.1
    num_inference_steps: int = 1000
    cubic: bool = False

class SDModelConfig(ModelConfig):
    unet: Path
    text_encoder: Path
    lda: Path

class IPAdapterConfig(ModelConfig):
    weights: Path | None = None
    embedding_dim: int = 768

class PaletteEncoderConfig(ModelConfig):
    weights: Path | None = None
    feedforward_dim: int = 3072
    num_attention_heads: int = 2 # 12, reduced for embedding_dim=768
    num_layers: int = 12
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    max_colors: int = 8
    mode : str = "transformer"
    weighted_palette: bool = False
    without_caption_probability: float = 0.17

class SpatialEncoderConfig(PaletteEncoderConfig):
    pass

class CustomHuggingfaceDatasetConfig(HuggingfaceDatasetConfig):
    caption_key: str = "caption"

class HistogramAutoEncoderConfig(ModelConfig):
    latent_dim : int = 64
    resnet_sizes: list[int] = [8, 8, 8, 8, 16, 16, 32]
    n_down_samples: int = 6
    color_bits: int = 6
    num_groups: int = 4
    loss: str = "kl_div"

CLIPFormatterLossModes = Literal["cross_attention", "cross_attention_query", "sum"]

class CLIPFormatterLossConfig(ModelConfig):
    embedding_dim: int = 768
    rank: int = 4
    scale: float = 0.0
    mode: CLIPFormatterLossModes = "sum"

class GenericEncoderConfig(ModelConfig):
    embedding_dim: int = 768
    num_layers: int = 0
    num_attention_heads: int = 2
    feedforward_dim: int = 20
    mode: str = 'transformer'

class Config(BaseConfig):
    latent_diffusion: LatentDiffusionConfig
    data: Path
    mode: Literal["text_embedding", "palette", "histogram", "pixel_sampling", "spatial_palette", "random_embedding", "image_embedding", "bw_image_embedding"]
    dataset: CustomHuggingfaceDatasetConfig
    eval_dataset: CustomHuggingfaceDatasetConfig
    grid_evaluation: GridEvaluationConfig
    mmd_evaluation: MmdEvaluationConfig
    clip_formatter_loss: CLIPFormatterLossConfig = CLIPFormatterLossConfig()
    offload_to_cpu: OffloadToCPUConfig = OffloadToCPUConfig()
    sd: SDModelConfig
    palette_encoder: PaletteEncoderConfig = PaletteEncoderConfig()
    spatial_palette_encoder: SpatialEncoderConfig = SpatialEncoderConfig()
    generic_encoder: GenericEncoderConfig = GenericEncoderConfig()
    ip_adapter: IPAdapterConfig = IPAdapterConfig()
    log_model_params: LogModelParamConfig = LogModelParamConfig()
    timestep_loss_rescaler: TimestepLossRescalerConfig = TimestepLossRescalerConfig()
    save_best_model: SaveBestModelConfig = SaveBestModelConfig()
    monitor_time: MonitorTimeConfig = MonitorTimeConfig()
    monitor_gradient: MonitorGradientConfig = MonitorGradientConfig()
    wandb: WandbConfig
    histogram_auto_encoder: HistogramAutoEncoderConfig = HistogramAutoEncoderConfig()
    visual_evaluation: VisualEvaluationConfig = VisualEvaluationConfig()
    fid_evaluation: FidEvaluationConfig = FidEvaluationConfig()

from pathlib import Path

from pydantic import BaseModel

from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.training_utils.wandb import WandbConfig
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig

from ip_adapter_palette.callback import LogModelParamConfig, MonitorTimeConfig, MonitorGradientConfig, OffloadToCPUConfig, TimestepLossRescalerConfig

class LatentDiffusionConfig(BaseModel):
    unconditional_sampling_probability: float = 0.2
    offset_noise: float = 0.1

class SDModelConfig(ModelConfig):
    unet: Path
    text_encoder: Path
    lda: Path

class IPAdapterConfig(ModelConfig):
    weights: Path | None = None

class PaletteEncoderConfig(ModelConfig):
    weights: Path | None = None
    feedforward_dim: int = 3072
    num_attention_heads: int = 12
    num_layers: int = 12
    embedding_dim: int = 768
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    max_colors: int
    mode : str = "transformer"
    weighted_palette: bool = False
    without_caption_probability: float = 0.17

class GridEvaluationConfig(BaseModel):
    db_indexes: list[int]
    batch_size: int = 1
    color_bits: int = 8
    seed: int = 0
    num_inference_steps: int = 30
    use_short_prompts: bool = False
    prompts: list[str] = []
    #num_images_per_prompt: int = 1
    condition_scale: float = 7.5

class MmdEvaluationConfig(BaseModel):
    batch_size: int = 1
    seed: int = 0
    num_inference_steps: int = 30
    condition_scale: float = 7.5

class CustomHuggingfaceDatasetConfig(HuggingfaceDatasetConfig):
    caption_key: str = "caption"

class Config(BaseConfig):
    latent_diffusion: LatentDiffusionConfig
    data: Path
    dataset: CustomHuggingfaceDatasetConfig
    eval_dataset: CustomHuggingfaceDatasetConfig
    grid_evaluation: GridEvaluationConfig
    # mmd_evaluation: MmdEvaluationConfig
    offload_to_cpu: OffloadToCPUConfig = OffloadToCPUConfig()
    sd: SDModelConfig
    palette_encoder: PaletteEncoderConfig
    ip_adapter: IPAdapterConfig = IPAdapterConfig()
    log_model_params: LogModelParamConfig = LogModelParamConfig()
    monitor_time: MonitorTimeConfig = MonitorTimeConfig()
    monitor_gradient: MonitorGradientConfig = MonitorGradientConfig()
    timestep_loss_rescaler: TimestepLossRescalerConfig = TimestepLossRescalerConfig()
    wandb: WandbConfig

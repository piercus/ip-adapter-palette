from pathlib import Path

from pydantic import BaseModel

from ip_adapter_lora.config import LatentDiffusionConfig, SDModelConfig
from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.training_utils.wandb import WandbConfig



class IPAdapterConfig(ModelConfig):
    weights: Path

class PaletteEncoderConfig(ModelConfig):
    weights: Path
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

class Config(BaseConfig):
    wandb: WandbConfig
    latent_diffusion: LatentDiffusionConfig
    data: Path
    offload_to_cpu: bool = False
    sd: SDModelConfig
    ip_adapter: IPAdapterConfig
    palette_encoder: PaletteEncoderConfig


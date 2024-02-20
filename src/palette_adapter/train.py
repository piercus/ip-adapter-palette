# from refiners.training_utils.config import (
#     OptimizerConfig,
#     Optimizers,
#     TrainingConfig,
#     LRSchedulerConfig,
#     LRSchedulerType
# )
# from ip_adapter_lora.config import Config, IPAdapterConfig, SDModelConfig, LatentDiffusionConfig

# from refiners.training_utils.wandb import WandbConfig

if __name__ == "__main__":
    # hub = Path("/home/trom/weights/")
    # sd_path = hub / "stable-diffusion-1-5/"
    # sd_config = SDModelConfig(
    #     unet=sd_path / "unet.safetensors",
    #     text_encoder=sd_path / "CLIPTextEncoderL.safetensors",
    #     lda=sd_path / "lda.safetensors",
    # )
    # ip_adapter_config = IPAdapterConfig(
    #     weights=hub / "IP-Adapter/ip-adapter-plus_sd15.safetensors",
    #     image_encoder_weights=hub / "stable-diffusion-2-1-unclip/CLIPImageEncoderH.safetensors",
    #     fine_grained=True,
    # )
    # training = TrainingConfig(
    #     duration="10_000:epoch",  # type: ignore
    #     batch_size=4,
    #     device="cuda:0",
    #     dtype="bfloat16",
    #     gradient_accumulation="16:step", # type: ignore
    #     evaluation_interval="2000:step",  # type: ignore
    # )
    # optimizer = OptimizerConfig(optimizer=Optimizers.AdamW8bit, learning_rate=2e-4)
    
    # scheduler = LRSchedulerConfig(
    #     type=LRSchedulerType.CONSTANT_LR,
    #     warmup="200:step",  # type: ignore
    # )

    # wandb = WandbConfig(
    #     notes="Testing latest changes.",
    #     mode="online",
    #     project="FG-1438-ip-adapter-lora",
    #     entity="finegrain",
    #     group="debug-runs",
    #     tags=["test"],
    #     #id="test-sd1-ip-adapter-no-lora-finegrained",
    # )

    # ld = LatentDiffusionConfig(
    #     unconditional_sampling_probability=0.2,
    #     offset_noise=0.1,
    # )

    config = Config.load_from_toml(config_file)

    trainer = SD1IPPalette(config)  # , callbacks=[OffloadToCPU(), SaveBestModel()])
    trainer.train()
from ip_adapter_palette.config import Config
from ip_adapter_palette.trainer import SD1IPPalette

if __name__ == "__main__":

    import sys

    config_path = sys.argv[1]
    config = Config.load_from_toml(config_path)

    trainer = SD1IPPalette(config)  # , callbacks=[OffloadToCPU(), SaveBestModel()])
    trainer.train()
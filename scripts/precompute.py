from ip_adapter_palette.config import Config
from ip_adapter_palette.trainer import SD1IPPalette
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config', help='path to the config file')
    parser.add_argument('--batch-size', type=int, default=1, help='an integer for the batch size')
    parser.add_argument('--force', action='store_true', help='If embeddings are already present, force recomputation')
    args = parser.parse_args()

    config = Config.load_from_toml(args.config)
    trainer = SD1IPPalette(config)
    trainer.precompute(batch_size=args.batch_size, force=args.force)
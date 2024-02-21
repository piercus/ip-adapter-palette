# ip-adapter-palette

Train a color palette adapter using color-palette

## Installation

### Prerequisites

Install [rye](https://github.com/mitsuhiko/rye)

### Install

```bash
git clone git@github.com:piercus/ip-adapter-palette.git
cd ip-adapter-palette
rye sync
```

### Download weights

We download the script from refiners, 

```bash
wget https://raw.githubusercontent.com/finegrain-ai/refiners/main/scripts/conversion/convert_transformers_clip_text_model.py -O scripts/convert_transformers_clip_text_model.py
wget https://raw.githubusercontent.com/finegrain-ai/refiners/main/scripts/conversion/convert_diffusers_autoencoder_kl.py -O scripts/convert_diffusers_autoencoder_kl.py
wget https://raw.githubusercontent.com/finegrain-ai/refiners/main/scripts/conversion/convert_diffusers_unet.py -O scripts/convert_diffusers_unet.py

```
And run them locally to download the weights
```bash
rye run python scripts/convert_transformers_clip_text_model.py --to weights/CLIPTextEncoderL.safetensors
rye run python scripts/convert_diffusers_autoencoder_kl.py --to weights/lda.safetensors
rye run python scripts/convert_diffusers_unet.py --to weights/unet.safetensors
```

## Train

### Prepare WandB

```
export WANDB_API_KEY=<key>
```

### Build the pre-computed embeddings

```
rye run python scripts/pre-compute.py configs/scheduled/finetune-color-palette-mlp-weighted.toml
```

### Run the training

NB : this is still WIP

```bash
rye run python scripts/train.py configs/scheduled/finetune-color-palette-mlp-weighted.toml
```


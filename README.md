# ip-adapter-palette

Train a color palette adapter using [refiners](https://github.com/finegrain-ai/refiners)

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

Download the scripts from refiners, 

```bash
# this commit should match with the one in pyproject.toml
export COMMIT=d199cd4f242ee33cff8ef9d6776bd171dac39434
wget https://raw.githubusercontent.com/finegrain-ai/refiners/${COMMIT}/scripts/conversion/convert_transformers_clip_text_model.py -O scripts/convert_transformers_clip_text_model.py
wget https://raw.githubusercontent.com/finegrain-ai/refiners/${COMMIT}/scripts/conversion/convert_diffusers_autoencoder_kl.py -O scripts/convert_diffusers_autoencoder_kl.py
wget https://raw.githubusercontent.com/finegrain-ai/refiners/${COMMIT}/scripts/conversion/convert_diffusers_unet.py -O scripts/convert_diffusers_unet.py
wget https://raw.githubusercontent.com/finegrain-ai/refiners/${COMMIT}/scripts/conversion/convert_transformers_clip_image_model.py -O scripts/convert_transformers_clip_image_model.py
wget https://raw.githubusercontent.com/finegrain-ai/refiners/${COMMIT}/scripts/conversion/convert_diffusers_ip_adapter.py -O scripts/convert_diffusers_ip_adapter.py
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin -O tmp/ip-adapter-plus_sd15.bin
```
And run them locally to download the weights
```bash
rye run python scripts/convert_transformers_clip_text_model.py --to weights/models/CLIPTextEncoderL.safetensors
rye run python scripts/convert_diffusers_autoencoder_kl.py --to weights/models/lda.safetensors
rye run python scripts/convert_diffusers_unet.py --to weights/models/unet.safetensors
rye run python scripts/convert_transformers_clip_image_model.py --to weights/models/CLIPImageEncoderH.safetensors
rye run python scripts/convert_diffusers_ip_adapter.py --from tmp/ip-adapter-plus_sd15.bin --to weights/models/IPAdapter.safetensors
```

## Train

### Prepare WandB

```bash
export WANDB_API_KEY=<key>
```

### Build the pre-computed embeddings

```bash
rye run python scripts/pre-compute.py configs/scheduled/finetune-color-palette-mlp-weighted.toml
```

### Run the training

NB : this is still WIP

```bash
rye run python scripts/train.py configs/scheduled/finetune-color-palette-mlp-weighted.toml
```

import torch
import os

# Directory path
directory = '/home/pierre/ip-adapter-palette/weights/embeddings/'

# Get all .pt files in the directory
files = [file for file in os.listdir(directory) if file.endswith('.pt')][0:10]

# Initialize variables
total_shape = 0
count = 0
print(len(files))

from refiners.fluxion.utils import summarize_tensor

def tensor_infos(tensor):
    return {
        "shape": tensor.shape,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean2": tensor.mean(dim=2),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "std2": tensor.std(dim=2).std(),
        "n_zeros": (tensor.abs() < 1e-8).sum().item(),
    }

# Iterate over each file
for file in files:
    # Load the torch file
    data = torch.load(os.path.join(directory, file))
    
    # Calculate the shape of data.text_embeddings
    print("source_text_embedding", tensor_infos(data['source_text_embedding'])["mean"])

    print("source_pixel_sampling", tensor_infos(data['source_pixel_sampling'])["mean"])




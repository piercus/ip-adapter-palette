import torch
import os

# Directory path
directory = '/home/pierre/ip-adapter-palette/weights/embeddings/'

# Get all .pt files in the directory
files = [file for file in os.listdir(directory) if file.endswith('.pt')]

# Initialize variables
total_shape = 0
count = 0
print(len(files))

# Iterate over each file
for file in files:
    # Load the torch file
    data = torch.load(os.path.join(directory, file))
    
    # Calculate the shape of data.text_embeddings
    shape = data['source_text_embeddings'].shape
    
    # Add the shape[2] value to the total_shape
    total_shape += shape[1]
    
    # Increment the count
    count += 1

# Calculate the average shape
average_shape = total_shape / count

# Print the average shape
print(f"The average shape of data.text_embeddings.shape[2] is: {average_shape}")

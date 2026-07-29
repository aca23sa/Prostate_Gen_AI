import os
import sys
import torch

# Import dataset class to scan through masks and find slices with tumors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.datasets.picai_dataset import PICAI2DDataset

# Load the validation dataset
print("Loading dataset")
val_dataset = PICAI2DDataset("data/splits/val.json", "data/processed/images", "data/processed/masks")

print(f"Scanning {len(val_dataset)} slices for tumor masks")

tumor_indices = []
for i in range(len(val_dataset)):
    sample = val_dataset[i]
    # If the maximum pixel value in the mask is greater than 0 then it contains a tumor
    if sample["mask"].max() > 0:
        tumor_indices.append(i)

print(f"Found {len(tumor_indices)} slices with visible tumors in the validation set.")
print("These are the first 30 indices available for the user_test.py script:")
print(tumor_indices[:30])

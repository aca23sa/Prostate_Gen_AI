import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from generative.networks.schedulers import DDPMScheduler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.multimodal_unet import MultiModalDiffusionUNet
from src.datasets.picai_dataset import PICAI2DDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# 1. Load Model & Scheduler
# ------------------
print("Loading model weights")
model = MultiModalDiffusionUNet(
    spatial_dims=2, in_channels=4, out_channels=3, cross_attention_dim=256 
).to(device)

checkpoint = torch.load("models/conditional/new_conditional_epoch_180.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0001, beta_end=0.02)
scheduler.set_timesteps(1000)

# ------------------
# 2. Load Dataset
# ------------------
val_dataset = PICAI2DDataset("data/splits/val.json", "data/processed/images", "data/processed/masks")

# ------------------
# 3. Interactive Loop
# ------------------
while True:
    user_input = input(f"Enter a slice index to test (0 to {len(val_dataset)-1}), or 'q' to quit: ")
    
    if user_input.lower() == 'q':
        print("Exiting test mode.")
        break
        
    try:
        idx = int(user_input)
        if idx < 0 or idx >= len(val_dataset):
            print("Index out of range. Try again.")
            continue
    except ValueError:
        print("Please enter a valid number.")
        continue

    # Fetch the specific slice
    sample = val_dataset[idx]
    
    # Add batch dimensions: [C, H, W] -> [1, C, H, W]
    real_tensor = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    clinical_data = sample["clinical"].unsqueeze(0).to(device)
    
    print(f"Generating synthetic MRI for Slice {idx}")
    if mask.max() == 0:
        print("Note: This slice has no tumor mask (Background only).")
    else:
        print("Tumor mask detected in this slice.")

    # Generate Fake Image 
    current_image = torch.randn((1, 3, 128, 128)).to(device) 
    with torch.no_grad():
        for t in scheduler.timesteps:
            model_input = torch.cat([current_image, mask], dim=1)
            noise_pred = model(model_input, torch.Tensor((t,)).to(device).long(), clinical_data)
            current_image, _ = scheduler.step(noise_pred, t, current_image)

    # Convert tensors to numpy arrays
    real_img = (real_tensor[0, 0].cpu().numpy() + 1) / 2      # T2 Channel
    fake_img = (current_image[0, 0].cpu().numpy() + 1) / 2    # T2 Channel
    mask_img = mask[0, 0].cpu().numpy()

    # Plotting the results side by side and saving the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mask_img, cmap='gray')
    axes[0].set_title(f"Input Mask (Blueprint)")
    axes[0].axis('off')
    
    axes[1].imshow(real_img, cmap='gray')
    axes[1].set_title("Real Patient T2")
    axes[1].axis('off')
    
    axes[2].imshow(fake_img.clip(0, 1), cmap='gray')
    axes[2].set_title("AI Generated T2")
    axes[2].axis('off')

    save_path = f"user_test_slice_{idx}.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"Download andopen '{save_path}' to see the results")
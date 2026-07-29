import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.multimodal_unet import MultiModalDiffusionUNet
from generative.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism

from src.datasets.picai_dataset import PICAI2DDataset

# ------------------
# Setup
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_determinism(42)

images_dir = "data/processed/images"
masks_dir = "data/processed/masks"  
train_split = "data/splits/train.json"

# Load the processed training dataset with associated tumour masks and clinical metadata.
train_dataset = PICAI2DDataset(train_split, images_dir=images_dir, masks_dir=masks_dir)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,       
    shuffle=True,
    num_workers=4,       
    pin_memory=True,     
    drop_last=True       
)

# ---------------------------------------------------------
# Conditional Diffusion Model Definition
# ---------------------------------------------------------
model = MultiModalDiffusionUNet(
    spatial_dims=2,
    in_channels=4,          # 3 MRI channels combined with 1 tumour mask channel
    out_channels=3,         # Predicting the noise for the 3 MRI channels
    cross_attention_dim=256 # Dimension used for conditioning with clinical features
).to(device)

# ------------------
# Diffusion Scheduler
# ------------------

# Defines the forward and reverse diffusion process used during training and image generation.
scheduler = DDPMScheduler(
    num_train_timesteps=1000,  
    schedule="linear_beta",
    beta_start=0.0001,
    beta_end=0.02,
)

# AdamW optimiser is used for stable diffusion training.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) 
# Mean squared error is used to compare predicted noise against sampled Gaussian noise.
criterion = nn.MSELoss()

num_epochs = 200

output_dir = "models/conditional"
os.makedirs(output_dir, exist_ok=True)

# ------------------
# Automatic Checkpoint Recovery
# ------------------
latest_path = os.path.join(output_dir, "latest_checkpoint.pth")
start_epoch = 0

if os.path.exists(latest_path):
    print(f"Loading conditional state from {latest_path}")
    checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming from Epoch {start_epoch}")
else:
    print("No conditional checkpoint found. Starting from scratch (Epoch 0).")

# ------------------
# Training Loop
# ------------------
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(train_loader):
    
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        clinical_data = batch["clinical"].to(device) 

        # Rescale image intensities from [0, 1] to [-1, 1] to match the diffusion model training distribution.
        images = images * 2.0 - 1.0

        # Sample noise for the images
        noise = torch.randn_like(images)

        # Sample timesteps randomly
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (images.shape[0],),
            device=device
        ).long()

        # Add noise to the clean MRI images
        noisy_images = scheduler.add_noise(images, noise, timesteps)


        model_input = torch.cat([noisy_images, masks], dim=1)

        # Predict the noise using the 4-channel input AND the clinical data
        noise_pred = model(x=model_input, timesteps=timesteps, clinical_data=clinical_data)

        # Calculate loss (Comparing 3-channel prediction to 3-channel actual noise)
        loss = criterion(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # Checkpoint Saving
    checkpoint_data = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }

    # Save the most recent training state after every epoch.
    torch.save(checkpoint_data, latest_path)
    print(f"Saved latest conditional checkpoint at epoch {epoch+1}")

    # Store additional milestone checkpoints every 10 epochs for long-term experiment tracking and evaluation.
    if (epoch + 1) % 10 == 0:
        milestone_path = os.path.join(output_dir, f"new_conditional_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), milestone_path)
        print(f"Saved historical milestone to {milestone_path}")
import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from generative.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.multimodal_unet import MultiModalDiffusionUNet
from src.datasets.picai_dataset import PICAI2DDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(42)

# ------------------
# 1. Create Output Folders
# ------------------
REAL_DIR = "evaluation/real_images"
FAKE_DIR = "evaluation/fake_images"
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# ------------------
# 2. Load Model & Scheduler
# ------------------
model = MultiModalDiffusionUNet(
    spatial_dims=2,
    in_channels=4,          
    out_channels=3,         
    cross_attention_dim=256 
).to(device)

weights_path = "models/conditional/latest_checkpoint.pth"
checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    beta_start=0.0001,
    beta_end=0.02,
)
scheduler.set_timesteps(1000)

# ------------------
# 3. Load Dataset
# ------------------
# Using val split to get images the model hasn't trained on
val_dataset = PICAI2DDataset(
    "data/splits/val.json", 
    images_dir="data/processed/images", 
    masks_dir="data/processed/masks"
)
# Batch size 1 makes it easier to track and save individual filenames
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ------------------
# 4. Helper Function to Save Images
# ------------------
def save_as_png(tensor_img, save_path):
    """ Converts a [-1, 1] tensor of shape (3, H, W) to a standard RGB PNG """
    # Squeeze out the batch dimension if present
    if len(tensor_img.shape) == 4:
        tensor_img = tensor_img[0]
        
    # Convert back to [0, 1] range
    img_np = (tensor_img / 2 + 0.5).clamp(0, 1).cpu().numpy()
    
    # Transpose from (C, H, W) to (H, W, C) for standard image formats
    img_np = img_np.transpose(1, 2, 0)
    
    # Convert to 0-255 pixel values
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    # Save using PIL
    Image.fromarray(img_uint8).save(save_path)

# ------------------
# 5. Mass Generation Loop
# ------------------
# The following loop iterates through the validation dataset
# and generates synthetic MRI slices for samples containing
# tumour masks. Both real and generated images are saved
# for subsequent FID evaluation and visual comparison.
NUM_TO_GENERATE = 100 
generated_count = 0

print(f"Starting mass generation for {NUM_TO_GENERATE} slices\n")

with torch.no_grad():
    for batch in val_loader:
        if generated_count >= NUM_TO_GENERATE:
            break
            
        real_tensor = batch["image"].to(device) # Shape: (1, 3, 256, 256)
        mask = batch["mask"].to(device)
        clinical_data = batch["clinical"].to(device)
        
        # Only evaluate slices that actually contain a tumor blueprint
        if mask.max() == 0:
            continue
            
        # Define the unique filename 
        filename = f"slice_{generated_count:04d}.png"
        
        # -------------------------------------------------
        # Save Real MRI Slice
        # -------------------------------------------------
        # The T2-weighted modality is selected for evaluation
        # as it provides the clearest anatomical representation
        # of the prostate structure.
        real_mri_np = real_tensor[0, 0].cpu().numpy() 

        real_uint8 = ((real_mri_np) * 255).astype(np.uint8)

        real_save_path = os.path.join(REAL_DIR, filename)
        Image.fromarray(real_uint8, mode='L').save(real_save_path)
        
        # -------------------------------------------------
        # Generate Synthetic MRI Slice
        # -------------------------------------------------
        # Random noise is progressively denoised using the
        # trained diffusion model conditioned on tumour masks
        # and associated clinical information.
        current_image = torch.randn((1, 3, 128, 128)).to(device) 
        
        for t in scheduler.timesteps:
            model_input = torch.cat([current_image, mask], dim=1)
            noise_pred = model(
                x=model_input, 
                timesteps=torch.Tensor((t,)).to(device).long(),
                clinical_data=clinical_data
            )
            current_image, _ = scheduler.step(noise_pred, t, current_image)
            
        # -------------------------------------------------
        # Save Generated MRI Slice
        # -------------------------------------------------
        # The generated T2-weighted channel is extracted
        # and converted into grayscale format for evaluation.
        fake_tensor = current_image[0, 0]
        fake_mri_np = (fake_tensor / 2 + 0.5).clamp(0, 1).cpu().numpy()
        fake_uint8 = (fake_mri_np * 255).astype(np.uint8)
        
        fake_save_path = os.path.join(FAKE_DIR, filename)
        Image.fromarray(fake_uint8, mode='L').save(fake_save_path)
        
        generated_count += 1
        if generated_count % 10 == 0:
            print(f"Generated {generated_count} / {NUM_TO_GENERATE} slices")

print(f"Saved 100 T2 grayscale images to '{REAL_DIR}' and '{FAKE_DIR}'.")
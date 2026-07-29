import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from generative.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism

# Add parent directory to path to import dataset and model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.multimodal_unet import MultiModalDiffusionUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
set_determinism(42)

# ------------------
# 1. Loading the Model
# ------------------
model = MultiModalDiffusionUNet(
    spatial_dims=2,
    in_channels=4,             # 3 (MRI) + 1 (Mask)
    out_channels=3,            # Predicting noise for 3 MRI channels
    cross_attention_dim=256    # Must match the MLP output dimension
).to(device)

# Load final epoch weights
weights_path = "models/conditional/latest_checkpoint.pth"
print(f"Loading weights from {weights_path}")

# Extract the weights if they were saved inside a dictionary
checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# ------------------
# 2. Setup Scheduler
# ------------------
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    beta_start=0.0001,
    beta_end=0.02,
)
scheduler.set_timesteps(num_inference_steps=1000)

# ------------------
# 3. Load a Validation Mask
# ------------------
with open("data/splits/val.json", "r") as f:
    val_files = json.load(f)

test_file = None
for fname in val_files:
    temp_mask_path = os.path.join("data/processed/masks", fname)
    temp_mask_np = np.load(temp_mask_path)
    if temp_mask_np.max() > 0:
        test_file = fname
        mask_np = temp_mask_np
        break

if test_file is None:
    print("Could not find a tumor in the val set, falling back to slice 0.")
    test_file = val_files[0]
    mask_np = np.load(os.path.join("data/processed/masks", test_file))
mask_tensor = torch.tensor(mask_np).unsqueeze(0).to(device) 

# ------------------
# 4. Define Clinical Data
# ------------------
# Must provide a clinical profile for the cross-attention conditioning. These values should be normalized (e.g., Min-Max scaling) based on the training data distribution.
# For example, this one is a 70-year-old patient with a PSA of 20, prostate volume of 50cc, and ISUP grade of 4 might be normalized to:
clinical_data = torch.tensor([[0.7, 0.8, 0.5, 0.9]], dtype=torch.float32).to(device)

# ------------------
# 5. Generate
# ------------------
print(f"Generating MRI for unseen mask: {test_file}")

current_image = torch.randn((1, 3, 128, 128)).to(device)

with torch.no_grad():
    for t in scheduler.timesteps:
        model_input = torch.cat([current_image, mask_tensor], dim=1)
        
        # Pass the clinical data into the forward pass
        noise_pred = model(
            x=model_input, 
            timesteps=torch.Tensor((t,)).to(device).long(),
            clinical_data=clinical_data
        )
        
        current_image, _ = scheduler.step(noise_pred, t, current_image)

# ------------------
# 6. Post-process and Save
# ------------------
final_image = (current_image / 2 + 0.5).clamp(0, 1).cpu().numpy()[0] 
mask_plot = mask_np[0]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(mask_plot, cmap="gray")
axes[0].set_title("Input Blueprint (Mask)")
axes[1].imshow(final_image[0], cmap="gray")
axes[1].set_title("Generated T2")
axes[2].imshow(final_image[1], cmap="gray")
axes[2].set_title("Generated ADC")
axes[3].imshow(final_image[2], cmap="gray")
axes[3].set_title("Generated HBV")

for ax in axes:
    ax.axis("off")

os.makedirs("results", exist_ok=True)
output_path = "results/final_multimodal_test.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Success! Saved generated image to {output_path}")
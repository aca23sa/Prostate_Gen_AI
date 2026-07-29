import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet

class MultiModalDiffusionUNet(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=4, out_channels=3, cross_attention_dim=256):
        """
        in_channels: e.g., 3 for noise + 1 for the tumor mask
        out_channels: e.g., 3 for the predicted noise of the 3 MRI channels
        cross_attention_dim: The size of our clinical embedding vector
        """
        super().__init__()
        
        # Multi-Layer Perceptron
        # Takes 4 clinical variables and projects them into a 256-dimensional space
        self.clinical_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(), # Standard activation function for diffusion models
            nn.Linear(64, cross_attention_dim)
        )
        
        # 2. The Standard MONAI U-Net
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(128, 256, 256),        
            attention_levels=(False, True, True), 
            num_res_blocks=1,
            num_head_channels=32,
            with_conditioning=True,              
            cross_attention_dim=cross_attention_dim # Matches MLP output
        )

    def forward(self, x, timesteps, clinical_data):
        """
        x: The noisy images (Batch, Channels, H, W)
        timesteps: The current timestep (Batch,)
        clinical_data: The 4 raw numbers from our dataloader (Batch, 4)
        """
        # Pass the 4 raw numbers through the MLP to get the embedding
        # Shape goes from (Batch, 4) -> (Batch, 256)
        clinical_embedding = self.clinical_mlp(clinical_data)
        
        # 2. MONAI's cross-attention expects a "Sequence" dimension
        # Since there is just one patient profile, adds an empty sequence dimension of 1
        # Shape becomes (Batch, 1, 256)
        clinical_embedding = clinical_embedding.unsqueeze(1)
        
        # 3. Pass everything into the U-Net
        return self.unet(x=x, timesteps=timesteps, context=clinical_embedding)
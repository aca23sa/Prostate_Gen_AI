import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd  

class PICAI2DDataset(Dataset):
    def __init__(self, split_file, images_dir="data/processed/images", masks_dir="data/processed/masks", transform=None, csv_path="data/raw/picai_labels/clinical_information/marksheet.csv"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        with open(split_file, "r") as f:
            self.slice_files = json.load(f)
        
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Extract the columns needed
        self.clinical_cols = ['patient_age', 'psa', 'prostate_volume', 'case_ISUP']
        df = df[['patient_id'] + self.clinical_cols].copy()
        
        # Fill any empty cells with the dataset median
        for col in self.clinical_cols:
            df[col] = df[col].fillna(df[col].median())
            
        # Normalization: Min-Max Scaling
        for col in self.clinical_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
            
        
        # Fast lookup dictionary using patient_id as the key
        df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True) 
        df.set_index('patient_id', inplace=True)
        self.clinical_dict = df.to_dict('index')

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        slice_filename = self.slice_files[idx]
        
        img_path = os.path.join(self.images_dir, slice_filename)
        mask_path = os.path.join(self.masks_dir, slice_filename)

        # Load numpy arrays
        img = np.load(img_path).astype(np.float32)   # shape (3, H, W)
        mask = np.load(mask_path).astype(np.float32) # shape (1, H, W)

        if self.transform:
            img = self.transform(img)
        
        # Extract the patient_id
        patient_id_str = slice_filename.split('_')[0]
        patient_id = int(patient_id_str)
        
        # Fetch the normalized variables from dictionary
        if patient_id in self.clinical_dict:
            patient_data = self.clinical_dict[patient_id]
            clinical_list = [
                patient_data['patient_age'],
                patient_data['psa'],
                patient_data['prostate_volume'],
                patient_data['case_ISUP']
            ]
        else:
            # If a patient is missing from the CSV, returns neutral 0.5 values
            clinical_list = [0.5, 0.5, 0.5, 0.5]
            
        # Convert list to a PyTorch tensor
        clinical_tensor = torch.tensor(clinical_list, dtype=torch.float32)
        
        return {
            "image": torch.from_numpy(img), 
            "mask": torch.from_numpy(mask),
            "clinical": clinical_tensor
        }
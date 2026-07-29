import os
import SimpleITK as sitk
import numpy as np
import json
import cv2
from sklearn.model_selection import train_test_split

# -------------------------
# Config
# -------------------------
RAW_DIR = "data/raw/picai"
PROCESSED_DIR = "data/processed/images"
SPLITS_DIR = "data/splits"
RAW_MASKS_DIR = "data/raw/picai_labels/csPCa_lesion_delineations/human_expert/resampled"
PROCESSED_MASKS_DIR = "data/processed/masks"

IMAGE_SIZE = 128
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(PROCESSED_MASKS_DIR, exist_ok=True)

def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

patients = sorted(os.listdir(RAW_DIR))
all_slices = []
patient_to_slices = {}

print("Starting preprocessing")

for patient_id in patients:
    patient_path = os.path.join(RAW_DIR, patient_id)

    t2_file = None
    adc_file = None
    hbv_file = None
    mask_file = None

    for f in os.listdir(patient_path):
        if f.endswith("_t2w.mha"):
            t2_file = os.path.join(patient_path, f)
        elif f.endswith("_adc.mha"):
            adc_file = os.path.join(patient_path, f)
        elif f.endswith("_hbv.mha"):
            hbv_file = os.path.join(patient_path, f)

    # Find the corresponding mask file
    if os.path.exists(RAW_MASKS_DIR):
        for f in os.listdir(RAW_MASKS_DIR):
            # This will match files like "10000_1000000.nii.gz" for patient "10000"
            if f.startswith(str(patient_id)) and f.endswith(".nii.gz"):
                mask_file = os.path.join(RAW_MASKS_DIR, f)
                break

    # Skip if any modality or the mask is missing
    if not (t2_file and adc_file and hbv_file and mask_file):
        continue

    # Load volumes
    t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_file))
    adc = sitk.GetArrayFromImage(sitk.ReadImage(adc_file))
    hbv = sitk.GetArrayFromImage(sitk.ReadImage(hbv_file))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))

    # Normalize
    t2 = normalize(t2)
    adc = normalize(adc)
    hbv = normalize(hbv)
    
    patient_slices = []

    num_slices = min(t2.shape[0], adc.shape[0], hbv.shape[0], mask.shape[0])

    for i in range(num_slices):
        slice_t2 = cv2.resize(t2[i], (IMAGE_SIZE, IMAGE_SIZE))
        slice_adc = cv2.resize(adc[i], (IMAGE_SIZE, IMAGE_SIZE))
        slice_hbv = cv2.resize(hbv[i], (IMAGE_SIZE, IMAGE_SIZE))

        # Resize mask using Nearest Neighbor to keep it binary
        slice_mask = cv2.resize(mask[i], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        if np.mean(slice_t2) < 0.05:
            continue

        # Stack into 3-channel image
        slice_img = np.stack([slice_t2, slice_adc, slice_hbv], axis=0)  # (3, H, W)

        # Expand mask to have a channel dimension: (1, H, W)
        slice_mask = np.expand_dims(slice_mask, axis=0).astype(np.float32)

        filename = f"{patient_id}_{i:03d}.npy"

        np.save(os.path.join(PROCESSED_DIR, filename), slice_img)
        np.save(os.path.join(PROCESSED_MASKS_DIR, filename), slice_mask)

        patient_slices.append(filename)
        all_slices.append(filename)

    patient_to_slices[patient_id] = patient_slices

print(f"Processed {len(patients)} patients.")
print(f"Total slices: {len(all_slices)}")

# -------------------------
# Create Splits (patient-level)
# -------------------------
patient_ids = list(patient_to_slices.keys())

trainval, test = train_test_split(
    patient_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train, val = train_test_split(
    trainval,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE
)

def expand_patients_to_slices(patient_list):
    slices = []
    for pid in patient_list:
        slices.extend(patient_to_slices[pid])
    return slices

train_slices = expand_patients_to_slices(train)
val_slices = expand_patients_to_slices(val)
test_slices = expand_patients_to_slices(test)

with open(os.path.join(SPLITS_DIR, "train.json"), "w") as f:
    json.dump(train_slices, f, indent=2)

with open(os.path.join(SPLITS_DIR, "val.json"), "w") as f:
    json.dump(val_slices, f, indent=2)

with open(os.path.join(SPLITS_DIR, "test.json"), "w") as f:
    json.dump(test_slices, f, indent=2)

print("Splits saved.")
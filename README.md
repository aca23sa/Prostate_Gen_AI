# GenAI Prostate MRI Synthesis using Conditional Diffusion Models

A multimodal conditional diffusion framework for synthetic prostate MRI generation using:

* T2-weighted MRI
* ADC MRI
* High b-value diffusion MRI (HBV)
* Tumor segmentation masks
* Clinical metadata (Age, PSA, ISUP Grade)

This repository accompanies the dissertation work on:

> *Conditional diffusion-based prostate MRI synthesis integrating spatial anatomical priors and non-spatial clinical metadata.*

---

# Project Overview

This project implements a conditional diffusion model based on a modified U-Net architecture for generating realistic prostate MRI slices.

The framework integrates:

* **Spatial conditioning**

  * Tumor segmentation masks

* **Temporal conditioning**

  * Diffusion timestep embeddings

* **Clinical conditioning**

  * Age
  * PSA
  * ISUP Grade Group

The generated outputs are evaluated using:

* Fréchet inception distance (FID)
* Qualitative visual assessment
* Anatomical consistency evaluation

---

# Repository Structure

```text
.
├── data/                           # Dataset directory (symlinked externally)
├── evaluation/
│   ├── fake_images/                # Generated synthetic MRI slices             
│   └── real_images/                # Real MRI slices for comparison
├── models/                         # Saved model checkpoints (symlinked externally)
├── results/                        # Dissertation result figures
├── scripts/
│   ├── find_tumors.py
│   ├── generate_conditional.py
│   ├── mass_evaluation.py
│   ├── train_picai_monai.py
│   └── user_test.py
├── slurm_scripts/                  # HPC SLURM job scripts
├── src/
│   ├── datasets/
│   │   ├── picai_dataset.py
│   │   └── preprocess_picai_2d.py
│   └── training/
│       └── multimodal_unet.py
└── README.md
```

---

# System Requirements

## Hardware

Recommended:

* NVIDIA GPU
* CUDA 11.8+
* ≥16GB VRAM preferred

Tested on:

* NVIDIA A100 GPUs
* Stanage HPC Cluster

---

# Installation

## 1. Clone the Repository

```bash
git clone [https://github.com/aca23sa/Prostate_Gen_AI.git]
cd Prostate_Gen_AI
```

---

## 2. Create a Python Virtual Environment

### Using venv

```bash
python3 -m venv venv
```

Activate the environment:

### Linux / HPC

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Dataset Acquisition

This project uses the PI-CAI prostate MRI dataset.

Dataset resources:

* https://pi-cai.grand-challenge.org/
* https://github.com/DIAGNijmegen/picai_baseline

---

# Downloading the Dataset

After obtaining access to the PI-CAI dataset:

Create the required directories:

```bash
mkdir -p data/raw/images
mkdir -p data/raw/labels
```

Place:

* MRI volumes into `data/raw/images`
* Segmentation masks into `data/raw/labels`

---

# Expected Data Structure

```text
data/
└── raw/
    ├── picai/
    │   ├── 10000/
    │   │   ├── 10000_1000000_adc.mha
    │   │   ├── 10000_1000000_hbv.mha
    │   │   ├── 10000_1000000_t2w.mha
    │   │   ├── 10000_1000000_cor.mha
    │   │   └── 10000_1000000_sag.mha
    │   ├── 10001/
    │   ├── 10002/
    │   └── ...
    └── picai_labels/
        ├── anatomical_delineations/
        ├── clinical_information/
        ├── csPCa_lesion_delineations/
        ├── additional_resources/
        ├── LICENSE
        └── README.md
```
Each patient folder inside picai/ contains multiple MRI modalities:

* adc → Apparent Diffusion Coefficient
* hbv → High b-value diffusion imaging
* t2w → T2-weighted MRI
* cor → Coronal view
* sag → Sagittal view

The picai_labels/ directory contains:

* anatomical_delineations/ → Anatomical prostate segmentation masks
* csPCa_lesion_delineations/ → Clinically significant prostate cancer lesion masks
* clinical_information/ → Patient and clinical metadata
* additional_resources/ → Supporting challenge resources

The preprocessing scripts automatically load the .mha MRI volumes and associated label masks for training and tumour-conditioned generation.

---

# Data Preprocessing

## Preprocess PI-CAI Data

Run:

```bash
python src/datasets/preprocess_picai_2d.py
```

This preprocessing pipeline performs:

* MRI normalization
* Slice extraction
* Modality alignment
* Tensor formatting
* Tumor mask integration

---

# Training the Diffusion Model

## Local Training

Run:

```bash
python scripts/train_picai_monai.py
```
The UNet architecture is implemented in:

```text
src/training/multimodal_unet.py
```
---

# Training on HPC using SLURM

Submit the training job:

```bash
sbatch slurm_scripts/train_gpu.sh
```

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=gen_ai_prostate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source venv/bin/activate

python scripts/train_picai_monai.py
```

---

# Generating Synthetic MRI Samples

Run:

```bash
python scripts/generate_conditional.py
```

Generated images will be saved in:

```text
results/
```

---

# Evaluating Results

# Mass Evaluation

To evaluate generated images against real MRI scans:

```bash
python scripts/mass_evaluation.py
```
Evaluation images are stored in:

```text
evaluation/
├── fake_images/
└── real_images/
```

Run FID evaluation:
```bash
python -m pytorch_fid evaluation/real_images evaluation/fake_images
```

# Visual Comparison

To identify and extract slices containing tumour regions:
```bash
python scripts/find_tumors.py
```
This script scans the validation dataset and outputs all MRI slice indices that contain tumour masks.

The output can be used to:

* Identify clinically significant tumour slices
* Select validation examples for qualitative analysis
* Compare tumour-conditioned generations against real MRI scans
* Support visual evaluation experiments

The generated slice indices can then be used with:
```bash
python scripts/user_test.py
```
user_test.py allows interactive visual comparison between:

* Real prostate MRI slices
* AI-generated prostate MRI slices
* Tumour-conditioned synthetic outputs
Users can input slice numbers identified by find_tumors.py to directly compare generated MRI outputs against the corresponding real MRI slices from the validation dataset.
---

# Clinical Conditioning

The diffusion model integrates:

* Age
* PSA
* ISUP Grade Group

through:

* MLP embedding layers
* Cross-attention conditioning
* Multimodal feature fusion

inside the conditional U-Net bottleneck and decoder blocks.

---

# Model Architecture

The architecture is based on:

* Conditional U-Net
* DDPM-style diffusion training
* Sinusoidal timestep embeddings
* Cross-attention conditioning

Input tensor:

```text
x_t ∈ R^(4 × 256 × 256)
```

Input channels:

1. T2 MRI
2. ADC MRI
3. HBV MRI
4. Tumor mask

---

# Reproducing Dissertation Results

## Step-by-Step Workflow

### 1. Clone the repository

```bash
git clone [https://github.com/aca23sa/Prostate_Gen_AI.git]
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and prepare PI-CAI dataset

Place MRI scans and masks into:

```text
data/raw/
```

### 5. Preprocess the dataset

```bash
python datasets/preprocess_picai_2d.py
```

### 6. Train the diffusion model

```bash
python scripts/train_picai_monai.py
```

### 7. Generate synthetic MRI images

```bash
python scripts/generate_conditional.py
```

### 8. Evaluate generated outputs

```bash
python scripts/mass_evaluation.py
```
FID Evaluation
The dissertation uses Fréchet Inception Distance (FID) to quantitatively compare generated MRI slices against real prostate MRI images.
After running:
```bash
python scripts/mass_evaluation.py
```
the generated and real images are placed into:
```text
evaluation/fake_images/
evaluation/real_images/
```
FID can then be computed using the Python FID package.
Install the package:
```bash
pip install pytorch-fid
```
Run FID evaluation:
```bash
python -m pytorch_fid evaluation/real_images evaluation/fake_images
```
Lower FID scores indicate that the generated images are more similar to the real MRI distribution.

### 9. Compare results

Compare newly generated outputs against the figures inside:
```text
results/
```

---

# Output Directories

Evaluation outputs:

```text
evaluation/
├── fake_images/
├── real_images/
```

Saved model checkpoints:

```text
models/conditional
```

---

# Notes

* Training from scratch may require several hours to days depending on GPU hardware.
* A100 GPUs are recommended for full-resolution diffusion training.
* Mixed precision training can significantly reduce VRAM usage.
* Running on HPC clusters is recommended for reproducibility.

---

# Citation

If you use this repository, please cite:

```bibtex
@misc{gen_ai_prostate,
  title={Conditional Diffusion Models for Synthetic Prostate MRI Generation},
  author={Shayaan Ather Hashmi},
  year={2026},
  url={https://github.com/aca23sa/Prostate_Gen_AI.git}
}
```

---

# Acknowledgements

* PI-CAI Challenge Dataset
* MONAI Framework
* PyTorch
* Hugging Face Diffusers

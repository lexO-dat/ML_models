# ML_models

This repository contains my utilities and custom implementations for various machine learning models, including both original architectures and those based on research papers.

## Current Uploaded Models

- **u-net/**: Implementation of the U-Net paper architecture for image segmentation.

- **Simple_GAN/**: Implementation of a GAN using only 1 layer nn (To show limitations and a brief idea of how GANs work).

- **DCGAN**: Implementation of a DCGAN with conv layers.

## Notes
- **Datasets** are referenced in code but should be downloaded separately due to size.
- **Dependencies** for each project are listed in their respective `requirements.txt` files.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/lexO-dat/ML_models.git
   ```
2. (Optional) Set up Git LFS for large model files:
   ```bash
   git lfs install
   git lfs pull
   ```
3. Install dependencies for the desired project:
   ```bash
   cd u-net
   pip install -r requirements.txt
   ```
4. Download datasets as needed (see code comments for dataset locations).

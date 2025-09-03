# FaceGen-VAE Project

This repository contains code, pre-trained models, and demo notebooks for reconstructing and enhancing face images using various Variational Autoencoder (VAE) architectures, including Vanilla VAE, StochasticVAE, BetaVAE, DFCVAE, and a PVAE (Probabilistic VAE) demo.

---

## 1️⃣ Requirements

All required Python packages are listed in **`requirements.txt`**.  

---

## 2️⃣ Required Files

Make sure you have all the necessary files in the correct paths:

- **Checkpoints (pre-trained models):**  
  - `vae_celeba_latent_200_epochs_10_batch_64_subset_80000.pth`  
  - `100.pt` (BetaVAE)  
  - `400.pt` (DFCVAE)  
- **Sample images for reconstruction**, e.g., `27994.jpg` in `Scripts/Images/`.

> Update the paths in the notebook/code to match where these files are on your system.

---

## 3️⃣ How to Run the Code

1. Open the notebook in **Google Colab**, **Jupyter Notebook**, or **VSCode**.  
2. Load all required files (code, images, checkpoints, `functions.py`, `model_classes`).  
3. Adjust the paths in the notebook for your system:  
   - `img_path`, `CKPT_PATH`, `checkpoint_path`, etc.  
4. Run each cell in order:  
   1. Import libraries and setup the environment  
   2. Check CUDA / device availability  
   3. Load pre-trained models (VAE, StochasticVAE, BetaVAE, DFCVAE)  
   4. Prepare input images and reconstruct them  
   5. Apply post-processing: enhancement, dehazing, smoothing, unsharp masking  
   6. Visualize results and compute MSE

> **Tip:** Using a GPU will significantly speed up reconstruction and generation. If only CPU is available, it will still run but slower.

---

## 4️⃣ Key Notes

- Each VAE model may have a different `latent_dim`. Make sure it matches the pretrained weights.  
- The helper functions in `functions.py` are required for image enhancement, dehazing, smoothing, and sharpening.  
- Multi-sample generation uses `with torch.no_grad()` to save memory and avoid gradient calculations.  
- The notebooks include side-by-side comparisons to visualize the effect of each processing step.

---

## 5️⃣ PVAE Image Reconstruction Demo

This Kaggle notebook demonstrates how to use a pre-trained **Probabilistic Variational Autoencoder (PVAE)** to reconstruct images from the CelebA dataset.

The notebook already includes:
- The pre-trained PVAE model.
- Sample test images for reconstruction.
- All necessary code to preprocess images, reconstruct them, add slight noise for variation, and visualize results.

You can view and run the notebook directly on Kaggle here:  
[PVAE Model on Kaggle](https://www.kaggle.com/code/samaahmedyousef/pvae-model)

### Features
- Reconstruct single images and compute the MSE against the original.
- Generate multiple noisy reconstructions to see variation.
- Display original and reconstructed images side by side.
- Sample new images from the PVAE latent space and visualize them in a grid.

---

## 6️⃣ Summary of Workflow

1. **Load libraries & check environment**  
2. **Load pre-trained models** (VAE variants)  
3. **Prepare images for input**  
4. **Reconstruct images using the model**  
5. **Apply enhancements**:  
   - Color/contrast adjustment  
   - Dehazing  
   - Smoothing + Unsharp Mask  
6. **Visualize results** side by side with MSE scores  
7. **Generate multiple samples** using stochastic sampling for variation  
8. **Compare models**: VAE, StochasticVAE, BetaVAE, DFCVAE, PVAE

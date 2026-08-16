# Project 1: Generative Modeling on ArtBench-10

Training and comparison of four generative approaches, a Variational AutoEncoder, a Generative Adversarial Network, a Denoising Diffusion Probabilistic Model, and a DDPM combined with a VAE, on the ArtBench-10 dataset of paintings across 10 artistic styles.

## Workflow
1. Load and explore the ArtBench-10 dataset and build PyTorch dataloaders.
2. Train the VAE, GAN, DDPM, and DDPM plus VAE models independently.
3. Run Bayesian hyperparameter optimization with Optuna, separately targeting the lowest training loss, FID, and KID for each model.
4. Retrain each model's best configuration on the full dataset and evaluate.
5. Compare generated samples visually and by FID and KID scores.

## Results
Among the diffusion based models, the plain DDPM achieved the best scores, outperforming the DDPM combined with a VAE. Visually, the GAN still produced sharper looking samples than the diffusion models at this scale, guiding further tuning of the DDPM's timestep count and model capacity.

## Technologies
Python, PyTorch, Optuna, Jupyter Notebook.

## Contents
* `project.ipynb` notebook with the project code
* `report.pdf` project report

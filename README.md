# Structural Analysis of the Spotify Tracks Dataset

Which audio features describe what a song sounds like, and which mostly add noise? This project studies about 114,000 Spotify tracks across 114 genres with several unsupervised methods. It then trains a small generative model to see whether that structure is enough to produce realistic synthetic songs.

*Rutgers, March–May 2026.*

## Approach

The notebook runs in six phases.

1. **Load and clean** the ten audio features: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo and duration.
2. **Distributions:** skew and kurtosis for each feature.
3. **Dependencies:** correlations and mutual information between the features, and between each feature and genre.
4. **Latent structure:** PCA, t-SNE and UMAP projections, plus a Gaussian mixture model with the number of clusters chosen by BIC (2 to 15 tried).
5. **Generative model:** a variational autoencoder (VAE) in PyTorch with a 3-dimensional latent space, small enough to plot and interpret directly.
6. **Synthetic data check:** compare generated songs with real ones using Kolmogorov–Smirnov (KS) tests and correlation structure.

## Findings

**One "intensity" axis dominates.** Three separate methods point to the same main source of variation: loud, energetic songs at one end and quiet, acoustic songs at the other.

| Method | What it shows |
|---|---|
| PCA | PC1 explains 28.7% of variance; its largest loadings are loudness (0.52), energy (0.51) and acousticness (−0.44). |
| VAE | The first latent dimension correlates with energy (r = 0.80), loudness (0.74) and acousticness (−0.69). |
| t-SNE / UMAP | Genres spread along the same gradient. |

**The clusters match real genres.** The Gaussian mixture model found six clusters. Four of them read clearly as musical groups:

| Cluster | Top genres |
|---|---|
| Quiet / instrumental | ambient, new-age, classical, piano |
| Heavy / loud | metalcore, grunge, heavy-metal, punk-rock |
| Speech / dance | comedy, j-dance, dancehall |
| Calm / background | sleep, study, idm |

**How realistic are the synthetic songs?** The VAE reproduces some features well and others poorly. A lower KS statistic means a closer match to real songs:

| Feature | KS statistic |
|---|---|
| Danceability | 0.11 (best) |
| Valence | 0.12 |
| Tempo | 0.17 |
| Speechiness | 0.36 |
| Liveness | 0.45 |
| Instrumentalness | 0.56 (worst) |

## Limitations

- **Partial mode collapse.** Rarer kinds of songs are underrepresented in the generated output. The model learns the common middle of the distribution better than its edges.
- **Spiky features are hard to model.** Instrumentalness and liveness are mostly near zero, with a small group of high values. The VAE's Gaussian latent space smooths these over, which explains their poor KS scores.
- **The differences are all statistically significant.** With 114,000 rows, every KS test rejects "same distribution" (p ≈ 0). The KS statistic itself, the size of the gap, is the more useful number here.

## How to run

**1. Get the data.** Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle and save it as `dataset.csv` next to the notebook. It is git-ignored, so it isn't in this repo.

**2. Install the dependencies.**

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn umap-learn torch
```

**3. Run `notebook.ipynb`** top to bottom. Figures and result tables are written to `figures/`, and the trained VAE is saved as `vae_model.pth`.

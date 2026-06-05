# Bird Clef 2025 - A top notebook from another competitor


He is wining in 2026 https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inferenceThis is not that notebook.

Here's a reverse-engineered breakdown of the training strategy, inferred from the inference code:
---

## Architecture: Sound Event Detection (SED)

The core is a SED model (CLEFClassifierSED) built from three components:
1. Mel Spectrogram frontend — 20-second audio chunks converted to 224×512 mel spectrograms (4096 n_fft, normalized with a per-sample z-score then min-max rescaling).
2. CNN backbone — a timm image classifier (features_only=True) used as a feature extractor — no classification head from pretrained weights.
3. AttHead — a custom temporal attention head using GeMFreq pooling (generalized mean over frequency), a 2-layer dense block, and a 1D conv for framewise logits. This outputs per-frame predictions across all 206 target classes.
The design is framewise, not clipwise — the model predicts at the 5-second segment level by processing the full 20s spectrogram and slicing it temporally.
---

## Multi-Model Ensemble

Seven models are ensembled, covering:
Model
Backbone
1
tf_efficientnet_b4.ns_jft_in1k
2
tf_efficientnet_b3.ns_jft_in1k
3 & 4
regnety_016.tv2_in1k
5
eca_nfnet_l0.ra2_in1k
6
regnety_008.pycls_in1k
7
tf_efficientnet_b0.ns_jft_in1k (Insects & Amphibians dataset)
They're blended with fixed non-uniform weights [0.133, 0.166, 0.133, 0.133, 0.166, 0.133, 0.133], giving slightly higher weight to models 2 and 5.
---

## Training Strategy (Reverse Engineered from Filenames)

The model weight filenames are highly informative:

### Self-Supervised / Pseudo-Label Loop

The key phrase iteration_3 and iteration_4 in filenames indicates iterative self-training (pseudo-labeling):
- An initial supervised model generates soft predictions on unlabeled data.
- Those predictions become pseudo-labels for the next training iteration.
- At least 4 rounds of this loop were run.

### Soft Label Temperature

temp_0.55 and temp_0.6 — pseudo-labels were temperature-scaled before use as training targets, sharpening the soft distributions.

### Augmentation

- MixUp with mixup_ratio=1, mixup_p=0.5 — aggressive input mixing during training.
- DropPath with drop_path_rate=0.15 — stochastic depth regularization in the backbone.

### Loss Function

ce in the filenames → Cross-Entropy loss (binary CE per class for multilabel).

### Cross-Validation

fold_2, fold_3 — k-fold cross-validation was used; each model corresponds to a different fold.

### Multi-Dataset Training

- Model 7 (incest_amphibia) was trained on a separate Insects & Amphibia dataset with 700 classes, then its predictions were remapped back to the 206 BirdCLEF target labels via a multilabel_to_train_labels dictionary.
- additional_data_full_data in one filename suggests augmentation with external/supplementary data for at least one model.
---

## Inference-Time Tricks

- Overlapping window inference (overlap_average_max_delta) — the 20s audio is slid with a 5s step, producing overlapping predictions that are averaged and then slightly time-shifted for a TTA-like delta correction.
- Gaussian temporal smoothing — a 5-tap kernel [0.1, 0.2, 0.4, 0.2, 0.1] smooths frame predictions over time.
- OpenVINO acceleration — backbone is compiled with Intel OpenVINO for fast CPU inference on Kaggle.
---
In short: iterative pseudo-labeling (4 rounds) + MixUp + CE loss + multi-backbone ensemble + a separate auxiliary model trained on a larger insect/amphibia dataset. Very systematic and execution-heavy.

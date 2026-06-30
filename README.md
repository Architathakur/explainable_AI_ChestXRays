# Explainable AI for Pneumonia Detection in Chest X-rays

This project detects pneumonia from frontal chest X-rays and explains the model's predictions using visual XAI methods.

The final working pipeline uses:

```text
RSNA Pneumonia Detection Challenge dataset
→ TorchXRayVision DenseNet121 encoder
→ cached 1024-dimensional image features
→ trained PyTorch MLP classification head
→ threshold tuning
→ Grad-CAM, Grad-CAM++, and Integrated Gradients explanations
```

Important clarification:

The repository contains an older ResNet50/TensorFlow path, but the final completed pipeline uses the **DenseNet121 + PyTorch + TorchXRayVision** path. The ResNet50 code is kept for reference only.

---

## Project Goal

The goal is to classify each chest X-ray as:

- `Normal`
- `Pneumonia`

and then answer an important medical AI question:

> When the model predicts pneumonia, is it looking at medically meaningful lung regions?

To study this, the project compares three explainability methods:

- Grad-CAM
- Grad-CAM++
- Integrated Gradients

---

## Dataset

The project uses the **RSNA Pneumonia Detection Challenge** dataset.

Expected local dataset files:

```text
rsna-pneumonia-detection-challenge/
├── stage_2_train_labels.csv
└── stage_2_train_images/
```

The dataset contains:

- DICOM chest X-ray images
- Binary pneumonia labels
- Bounding boxes for many pneumonia-positive cases

The dataset itself is not committed to Git because it is large.

### Data Split

The local split used in this project:

| Split | Images | Pneumonia | Normal |
|---|---:|---:|---:|
| Train | 18,678 | 4,208 | 14,470 |
| Validation | 4,003 | 902 | 3,101 |
| Test | 4,003 | 902 | 3,101 |

Approximate class balance:

| Class | Percentage |
|---|---:|
| Normal | 78% |
| Pneumonia | 22% |

This imbalance is why focal loss, balanced batches, and threshold tuning are useful.

---

## Final Model

The final model uses a frozen TorchXRayVision DenseNet121 encoder:

```text
densenet121-res224-all
```

This encoder was pretrained on multiple public chest X-ray datasets, making it more suitable for medical X-rays than a generic ImageNet-only backbone.

The classification architecture is:

```text
Chest X-ray
→ DenseNet121 CXR encoder
→ 1024-dimensional feature vector
→ MLP head
→ pneumonia probability
```

Final MLP head configuration:

| Setting | Value |
|---|---:|
| Hidden dimension | 512 |
| Dropout | 0.4 |
| Loss | Balanced focal loss |
| Optimizer | Adam |
| Early stopping metric | Validation F1 |
| Backend | PyTorch MPS on Mac, CPU fallback |

---

## Final Validation Results

The final saved checkpoint is:

```text
outputs/cxr_foundation_densenet121-res224-all/best_cxr_foundation.pt
```

The best threshold found on the validation split was:

```text
0.731
```

Validation metrics after threshold tuning:

| Metric | Value |
|---|---:|
| Accuracy | 0.8294 |
| Precision | 0.6081 |
| Recall | 0.6829 |
| F1 | 0.6433 |
| AUC-ROC | 0.8731 |
| AUC-PR | 0.6809 |

Confusion matrix:

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 2,704 | 397 |
| Actual Pneumonia | 286 | 616 |

### Interpretation

The model achieves a strong AUC-ROC of about `0.87`, which means it ranks pneumonia cases above normal cases well across thresholds.

The F1 score is about `0.64`, which is reasonable for an imbalanced medical imaging classification task using a frozen encoder and lightweight head.

Compared with an ideal research-grade project:

| Area | This project | Strong/ideal project |
|---|---:|---:|
| AUC-ROC | 0.873 | 0.90+ |
| F1 | 0.643 | 0.70+ |
| External test set | Not included | Yes |
| Radiologist comparison | Not included | Yes |
| Calibration analysis | Basic threshold tuning | Full calibration study |
| XAI evaluation | IoU, time, consistency | IoU plus expert review |

So the project is solid for an academic/deep-learning portfolio project, but not a clinical deployment system.

---

## XAI Results

The project implemented three XAI methods:

### Grad-CAM

Uses gradients from the final DenseNet convolutional block to highlight spatial regions important to the prediction.

Target layer:

```text
features.denseblock4
```

### Grad-CAM++

An extension of Grad-CAM that uses higher-order gradient weighting. It can produce sharper localization in some cases.

### Integrated Gradients

Uses Captum to attribute the prediction back to input pixels by comparing the X-ray against a black baseline.

Final Integrated Gradients settings:

| Setting | Value |
|---|---:|
| Baseline | Zero tensor / black image |
| Steps | 50 for full method, lower values used for smoke testing |
| Method | `gausslegendre` |

### Quantitative XAI Comparison

Saved result file:

```text
outputs/xai_quantitative.json
```

| Method | Mean IoU | Time Mean | Time Std | Consistency |
|---|---:|---:|---:|---:|
| Grad-CAM | 0.2252 | 455.85 ms | 267.74 ms | 1.0000 |
| Grad-CAM++ | 0.2825 | 537.46 ms | 336.11 ms | 1.0000 |
| Integrated Gradients | 0.1888 | 2255.49 ms | 715.60 ms | 1.0000 |

Interpretation:

- Grad-CAM++ had the best mean IoU in this run.
- Grad-CAM was faster than Grad-CAM++.
- Integrated Gradients was much slower because it requires many forward/backward integration steps.
- All methods were deterministic and had near-perfect consistency.

---

## Generated Figures

The final figures are saved in:

```text
outputs/figures/
```

Expected figures:

| File | Description |
|---|---|
| `metrics_table.png` | Classification metrics table |
| `confusion_matrix.png` | Confusion matrix |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision-recall curve |
| `xai_comparison_pneumonia.png` | Correct pneumonia cases with all XAI methods |
| `xai_comparison_normal.png` | Correct normal cases with all XAI methods |
| `xai_failure_cases.png` | Misclassified cases with XAI overlays |
| `xai_quantitative.png` | Bar chart comparing XAI IoU and runtime |

Individual case-level XAI overlays are saved in:

```text
outputs/figures/cases/
```

---

## Code Structure

```text
xai_pneumonia/
├── data/
│   ├── data_loader.py
│   └── splits.json
├── model/
│   ├── cxr_torch_model.py
│   └── resnet50_model.py
├── xai/
│   ├── explain.py
│   ├── common.py
│   ├── gradcam.py
│   ├── gradcam_plus_plus.py
│   ├── integrated_gradients.py
│   └── layer_selector.py
├── evaluation/
│   ├── metrics.py
│   ├── statistical_tests.py
│   └── visualizer.py
├── notebooks/
│   └── full_pipeline.ipynb
├── train_cxr_foundation.py
├── tune_threshold.py
├── train.py
├── evaluate.py
└── requirements-mac-training.txt
```

### Most Important Final Files

| File | Purpose |
|---|---|
| `xai_pneumonia/data/data_loader.py` | Loads RSNA labels, DICOMs, bounding boxes, and split metadata |
| `xai_pneumonia/model/cxr_torch_model.py` | Defines the TorchXRayVision DenseNet encoder and MLP head |
| `xai_pneumonia/train_cxr_foundation.py` | Main DenseNet/PyTorch training and feature caching script |
| `xai_pneumonia/tune_threshold.py` | Finds the best validation threshold and prints metrics |
| `xai_pneumonia/xai/explain.py` | Generates Grad-CAM, Grad-CAM++, Integrated Gradients, figures, and XAI metrics |
| `xai_pneumonia/requirements-mac-training.txt` | Mac/PyTorch/MPS training dependencies |

### Legacy Files

| File | Status |
|---|---|
| `xai_pneumonia/train.py` | Older TensorFlow/ResNet50 training path |
| `xai_pneumonia/model/resnet50_model.py` | Older ResNet50 model |
| `xai_pneumonia/evaluate.py` | Older evaluation entry point |
| `xai_pneumonia/notebooks/full_pipeline.ipynb` | Mostly documents the older notebook workflow |

---

## Setup on Mac

Create a virtual environment:

```bash
python -m venv .venv-train
source .venv-train/bin/activate
```

Install dependencies:

```bash
pip install -r xai_pneumonia/requirements-mac-training.txt
```

Verify PyTorch MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Expected output on Apple Silicon with MPS support:

```text
True
```

---

## Running the Final Pipeline

The dataset should already exist locally. Do not redownload it unless needed.

### 1. Train the DenseNet MLP Head

```bash
PYTHONPATH=. .venv-train/bin/python -m xai_pneumonia.train_cxr_foundation \
  --data_dir rsna-pneumonia-detection-challenge \
  --weights densenet121-res224-all \
  --output_dir outputs/cxr_foundation_densenet121-res224-all \
  --feature_cache_dir outputs/feature_cache \
  --hidden_dim 512 \
  --dropout 0.4 \
  --loss focal \
  --device mps
```

This script:

1. Loads RSNA DICOM images.
2. Extracts DenseNet features.
3. Caches train and validation features.
4. Trains the MLP head.
5. Saves the best checkpoint.

### 2. Tune the Threshold

```bash
PYTHONPATH=. .venv-train/bin/python -m xai_pneumonia.tune_threshold \
  --checkpoint outputs/cxr_foundation_densenet121-res224-all/best_cxr_foundation.pt \
  --features outputs/feature_cache/densenet121-res224-all_val_full.npz \
  --output_dir outputs/threshold_tuning_densenet121-res224-all
```

This finds the probability threshold with the best validation F1 score.

### 3. Generate XAI Results

```bash
PYTHONPATH=. .venv-train/bin/python -m xai_pneumonia.xai.explain \
  --data_dir rsna-pneumonia-detection-challenge \
  --checkpoint outputs/cxr_foundation_densenet121-res224-all/best_cxr_foundation.pt \
  --threshold_json outputs/threshold_tuning_densenet121-res224-all/threshold_tuning.json \
  --output_dir outputs
```

This generates:

1. Grad-CAM overlays.
2. Grad-CAM++ overlays.
3. Integrated Gradients overlays.
4. Case comparison figures.
5. Quantitative XAI metrics.

---

## Historical Experiments

Several other model ideas were explored before settling on the final DenseNet/PyTorch pipeline.

### Original ResNet50 Baseline

The original path used:

```text
224x224x3 input
→ ImageNet ResNet50 backbone
→ GlobalAveragePooling
→ Dense layers
→ softmax classification
```

This path remains in:

```text
xai_pneumonia/train.py
xai_pneumonia/model/resnet50_model.py
```

It is not the final model.

### TorchXRayVision Encoder Comparison

Several frozen TorchXRayVision DenseNet encoders were compared. The `densenet121-res224-all` checkpoint performed best overall.

### Short Fine-Tuning

Short fine-tuning of DenseNet layers was explored, but it did not meaningfully improve results over the frozen encoder.

### MIL / Attention Experiments

Spatial MIL and attention-based approaches were explored for better localization. They were useful conceptually, but the final reproducible pipeline is the frozen DenseNet encoder plus MLP head.

---

## Notes

- Large files are intentionally excluded from Git: DICOM images, feature caches, checkpoints, virtual environments, and generated output folders.
- The reported numbers are validation results from the local split.
- The project is for research and education, not clinical use.
- For final project understanding, read the documentation files before the notebook because the notebook includes older ResNet/TensorFlow material.


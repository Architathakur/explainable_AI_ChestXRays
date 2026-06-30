# XAI Pneumonia Chest X-Ray Project: A-to-Z Interview Guide

## 1. Project in One Line

This project classifies pneumonia from RSNA chest X-ray DICOM images using a frozen TorchXRayVision DenseNet121 medical-imaging encoder and a trained PyTorch MLP classifier head, then explains the model decisions using Grad-CAM, Grad-CAM++, and Integrated Gradients.

## 2. Problem Statement

The goal is binary classification:

- Class `0`: Normal / no pneumonia opacity.
- Class `1`: Pneumonia.

The project also addresses explainability. In medical AI, accuracy alone is not enough. Doctors and evaluators need to understand where the model is looking. Therefore, the project generates heatmap overlays and evaluates whether the highlighted regions overlap with RSNA bounding boxes.

## 3. Dataset

Dataset used:

- RSNA Pneumonia Detection Challenge.
- Image format: DICOM chest X-rays.
- Label file: `stage_2_train_labels.csv`.
- Image folder: `stage_2_train_images/`.

The dataset was already present locally. No dataset download was performed.

Verified split:

| Split | Count | Pneumonia | Normal | Pneumonia Ratio |
|---|---:|---:|---:|---:|
| Train | 18,678 | 4,208 | 14,470 | 22.53% |
| Validation | 4,003 | 902 | 3,101 | 22.53% |
| Test | 4,003 | 902 | 3,101 | 22.53% |

This confirms class imbalance: roughly 22% pneumonia and 78% normal.

## 4. High-Level Pipeline

The final pipeline is:

1. Read RSNA DICOM image.
2. Convert grayscale pixel data to normalized 224 x 224 tensor.
3. Feed tensor into TorchXRayVision DenseNet121 encoder.
4. Extract 1024-dimensional feature vector.
5. Cache features to disk.
6. Train an MLP classification head on cached features.
7. Tune decision threshold on validation set.
8. Generate classification metrics and figures.
9. Generate XAI heatmaps using Grad-CAM, Grad-CAM++, and Integrated Gradients.
10. Quantitatively compare XAI methods using localization IoU, runtime, and consistency.

## 5. Why DenseNet121 and Not ResNet50

The codebase had an older TensorFlow/ResNet50 path, but that path was abandoned for this run.

Final model path:

- PyTorch.
- TorchXRayVision DenseNet121.
- Weights: `densenet121-res224-all`.
- Encoder frozen.
- MLP head trained.

Reason:

TorchXRayVision provides models pretrained on large chest X-ray datasets, which is more domain-appropriate than using a generic ImageNet model. A frozen medical encoder is also efficient on a MacBook because we only train a small classifier head.

## 6. Environment Setup

A virtual environment was created:

```bash
python3 -m venv .venv-train
```

Dependencies installed included:

- `torch`
- `torchvision`
- `torchxrayvision`
- `captum`
- `grad-cam`
- `scikit-learn`
- `matplotlib`
- `pydicom`
- `opencv-python`
- `tqdm`
- `numpy`
- `pandas`

The requirements file updated:

- `xai_pneumonia/requirements-mac-training.txt`

MPS was verified outside the sandbox during training:

```text
Using device: mps
```

## 7. Important Code Fixes Made

Several issues had to be fixed before the pipeline could run.

### 7.1 TensorFlow Imports Removed From PyTorch Path

The project had many TensorFlow imports in shared modules and `__init__.py` files. These broke the PyTorch-only training environment because TensorFlow was intentionally not installed.

Fix:

- Removed or lazy-loaded TensorFlow imports in shared modules.
- Kept legacy TensorFlow code untouched where possible, but prevented it from blocking PyTorch imports.

### 7.2 Project Root Fixed

`get_project_root()` originally pointed to the package folder instead of the project root. This caused outputs to be written under the wrong directory.

Fix:

- Updated it to resolve to the root folder:

```text
xai-pneumonia-cxr-main/
```

### 7.3 TorchXRayVision Cache Redirected

TorchXRayVision tried to write pretrained weights into:

```text
/Users/archita/.torchxrayvision
```

This caused permission/sandbox problems.

Fix:

- Redirected pretrained weight cache to:

```text
outputs/torchxrayvision/
```

### 7.4 Feature Extraction Progress Added

Full DICOM feature extraction was slow and initially silent.

Fix:

- Added `tqdm` progress bars.
- Added `--num_workers` to use multiple DataLoader workers.

Final feature extraction used:

```bash
--num_workers 4
```

### 7.5 XAI Rewritten for DenseNet/PyTorch

The original XAI entrypoint was TensorFlow/ResNet-based.

Fix:

- Added new DenseNet/PyTorch XAI implementation:

```text
xai_pneumonia/xai/explain.py
```

It implements:

- Grad-CAM.
- Grad-CAM++.
- Integrated Gradients.
- Side-by-side overlays.
- XAI quantitative metrics.
- Figure generation.

### 7.6 Frozen Encoder Gradient Issue Fixed

Grad-CAM requires gradients through the feature map. Because the DenseNet encoder is frozen, the activation tensors did not automatically require gradients.

Fix:

- Enabled gradients on the explanation input tensor during Grad-CAM and Grad-CAM++ generation.

## 8. Model Architecture

### 8.1 Encoder

Encoder:

```text
TorchXRayVision DenseNet121
weights = densenet121-res224-all
```

Input:

```text
1 x 224 x 224 chest X-ray tensor
```

Output:

```text
1024-dimensional feature vector
```

The encoder was frozen, meaning its pretrained weights were not updated.

### 8.2 Classifier Head

The trained head is an MLP:

```text
Linear(1024 -> 512)
ReLU
Dropout(0.4)
Linear(512 -> 2)
```

This outputs logits for:

- Normal.
- Pneumonia.

## 9. Image Preprocessing

For each DICOM:

1. Read using `pydicom`.
2. Extract pixel array.
3. Handle `MONOCHROME1` images by inverting intensities.
4. Normalize pixel values to `[0, 1]`.
5. Resize to `224 x 224`.
6. Convert to TorchXRayVision-style input:

```text
image = (2 * image - 1) * 1024
```

This maps pixel intensity into approximately:

```text
[-1024, 1024]
```

## 10. Feature Caching

Feature caching was performed to avoid repeatedly passing all images through DenseNet during MLP training.

Saved files:

```text
outputs/feature_cache/densenet121-res224-all_train_full.npz
outputs/feature_cache/densenet121-res224-all_val_full.npz
```

Feature shapes:

| Split | Shape |
|---|---|
| Train | 18,678 x 1024 |
| Validation | 4,003 x 1024 |

This made head training very fast after the first extraction.

## 11. Training Setup

Training command used conceptually:

```bash
.venv-train/bin/python -u -m xai_pneumonia.train_cxr_foundation \
  --epochs 30 \
  --early_stopping_patience 6 \
  --batch_size 64 \
  --feature_batch_size 32 \
  --balanced_sampler \
  --num_workers 4 \
  --device auto
```

Training settings:

| Setting | Value |
|---|---|
| Encoder | Frozen DenseNet121 |
| Head hidden dimension | 512 |
| Dropout | 0.4 |
| Loss | Balanced focal loss |
| Optimizer | AdamW |
| Early stopping | Validation F1 |
| Device | MPS |

Checkpoint saved:

```text
outputs/cxr_foundation_densenet121-res224-all/best_cxr_foundation.pt
```

## 12. Why Focal Loss Was Used

The dataset is imbalanced:

- Normal: about 78%.
- Pneumonia: about 22%.

If standard cross-entropy is used naively, the model can become biased toward predicting normal.

Balanced focal loss helps by:

- Giving more importance to difficult examples.
- Reducing the dominance of easy majority-class samples.
- Improving learning for pneumonia cases.

## 13. Threshold Tuning

The raw model output gives probabilities. A default threshold of `0.5` was too recall-heavy and produced many false positives.

Threshold tuning searched thresholds and selected the one with best pneumonia F1.

Best threshold:

```text
0.731
```

Why threshold tuning matters:

- Medical datasets are imbalanced.
- The best operating point is rarely exactly 0.5.
- For clinical screening, one may prefer higher recall.
- For reliable diagnosis support, one may prefer balanced F1 or higher precision.

## 14. Classification Results

### 14.1 Default Threshold 0.50

| Metric | Value |
|---|---:|
| Accuracy | 0.5536 |
| Precision | 0.3332 |
| Recall | 0.9800 |
| F1 | 0.4973 |
| AUC-ROC | 0.8731 |
| AUC-PR | 0.6809 |

Confusion matrix:

```text
[[1332, 1769],
 [  18,  884]]
```

Interpretation:

- Very high recall.
- Very few missed pneumonia cases.
- Too many false positives.
- Good for aggressive screening, but not balanced.

### 14.2 Best F1 Threshold 0.731

| Metric | Value |
|---|---:|
| Accuracy | 0.8294 |
| Precision | 0.6081 |
| Recall | 0.6829 |
| F1 | 0.6433 |
| AUC-ROC | 0.8731 |
| AUC-PR | 0.6809 |

Confusion matrix:

```text
[[2704, 397],
 [ 286, 616]]
```

Interpretation:

- Much better balance than threshold 0.5.
- Accuracy improves substantially.
- Precision improves from 0.3332 to 0.6081.
- Recall decreases from 0.9800 to 0.6829.
- F1 improves from 0.4973 to 0.6433.

### 14.3 High-Recall Threshold 0.679

| Metric | Value |
|---|---:|
| Accuracy | 0.7787 |
| Precision | 0.5056 |
| Recall | 0.8016 |
| F1 | 0.6201 |

Interpretation:

This is useful if the system is positioned as a screening tool where missing pneumonia is more costly than over-alerting.

### 14.4 Higher-Precision Threshold 0.755

| Metric | Value |
|---|---:|
| Accuracy | 0.8366 |
| Precision | 0.6520 |
| Recall | 0.5898 |
| F1 | 0.6193 |

Interpretation:

This is useful if we want fewer false positives, but it misses more pneumonia cases.

## 15. How Good Are These Results?

### 15.1 Your Result Summary

The key result is:

```text
AUC-ROC = 0.8731
Best F1 = 0.6433
Accuracy = 0.8294
Precision = 0.6081
Recall = 0.6829
```

This is a solid baseline for:

- Frozen medical foundation encoder.
- Simple MLP head.
- No full fine-tuning of DenseNet.
- Running on a MacBook.
- Class-imbalanced RSNA dataset.

### 15.2 Comparison With a Good Project

For a good academic/class project:

| Metric | Good Target | Your Result | Comment |
|---|---:|---:|---|
| AUC-ROC | 0.85 to 0.90 | 0.8731 | Good |
| F1 | 0.60 to 0.70 | 0.6433 | Good |
| Accuracy | 0.80 to 0.88 | 0.8294 | Good |
| Precision | 0.60 to 0.75 | 0.6081 | Acceptable |
| Recall | 0.65 to 0.80 | 0.6829 | Acceptable |
| AUC-PR | 0.60 to 0.75 | 0.6809 | Good |

Conclusion:

Your model is good for a project-level implementation. It meets the expected success criteria:

- AUC-ROC is above 0.87.
- F1 is above 0.64 after threshold tuning.

### 15.3 Comparison With an Ideal Research-Level Project

For a stronger research-grade project, expected targets may be:

| Metric | Strong / Ideal Target | Your Result | Gap |
|---|---:|---:|---|
| AUC-ROC | 0.90 to 0.95+ | 0.8731 | Needs fine-tuning / ensembling |
| F1 | 0.70 to 0.80+ | 0.6433 | Needs better calibration and training |
| Precision | 0.70 to 0.85 | 0.6081 | False positives still high |
| Recall | 0.75 to 0.90 | 0.6829 | Some pneumonia cases missed |
| AUC-PR | 0.75 to 0.85+ | 0.6809 | Needs improved minority-class ranking |

To reach ideal results, the next improvements would be:

- Fine-tune the last DenseNet block.
- Use stronger augmentations.
- Try calibration methods such as temperature scaling.
- Try class-balanced batch sampling plus focal loss tuning.
- Use ensemble models.
- Train longer with learning-rate scheduling.
- Use bounding-box-aware weak supervision if localization labels are available.

## 16. XAI Methods Implemented

Three methods were implemented.

### 16.1 Grad-CAM

Grad-CAM uses gradients flowing into a convolutional layer to identify important spatial regions.

Target layer:

```text
DenseNet features.denseblock4
```

Steps:

1. Forward pass image through model.
2. Select predicted class score.
3. Compute gradient of class score with respect to denseblock4 feature maps.
4. Global-average-pool gradients over spatial dimensions.
5. Use pooled gradients as channel weights.
6. Weighted sum feature maps.
7. Apply ReLU.
8. Resize to 224 x 224.
9. Overlay on original X-ray using jet colormap.

Interview explanation:

Grad-CAM tells us which image regions most influenced the model's class decision by using gradients from the last convolutional feature maps.

### 16.2 Grad-CAM++

Grad-CAM++ improves Grad-CAM by using higher-order gradient weighting.

Why useful:

- Can localize multiple relevant regions better than standard Grad-CAM.
- Often gives sharper maps.

Implementation:

- Uses `pytorch-grad-cam` if available.
- Falls back to manual Grad-CAM++ if needed.

### 16.3 Integrated Gradients

Integrated Gradients attributes prediction importance back to input pixels.

Baseline:

```text
black image / zero tensor
```

Concept:

Instead of only looking at one gradient, IG integrates gradients along a path from a baseline image to the actual image.

Formula idea:

```text
Attribution = (input - baseline) * average_gradient_along_path
```

In this project:

- Default implementation supports `n_steps=50`.
- Final batch run used `ig_steps=8` on CPU for safety after a Mac kernel panic during MPS XAI.

## 17. Why XAI Was Run on CPU

During an earlier XAI smoke test on MPS, the Mac restarted due to a kernel watchdog panic:

```text
watchdog timeout: no checkins from watchdogd
```

This was a macOS/kernel-level issue, not a normal Python exception.

To avoid another restart:

- Training and feature extraction used MPS.
- XAI was run on CPU.
- Integrated Gradients was run with fewer steps for the final batch.

This was a practical engineering tradeoff.

## 18. XAI Quantitative Results

| Method | Mean IoU | Time Mean | Time Std | Consistency |
|---|---:|---:|---:|---:|
| Grad-CAM | 0.2252 | 455.8 ms | 267.7 ms | 1.0000 |
| Grad-CAM++ | 0.2825 | 537.5 ms | 336.1 ms | 1.0000 |
| Integrated Gradients | 0.1888 | 2255.5 ms | 715.6 ms | 1.0000 |

Interpretation:

- Grad-CAM++ had the best localization IoU.
- Integrated Gradients was slowest.
- All methods were deterministic, giving consistency approximately 1.0.

## 19. What Is Mean IoU in XAI Evaluation?

IoU means Intersection over Union.

In this project:

1. Take the XAI heatmap.
2. Select top 20% most activated pixels.
3. Convert that into a binary mask.
4. Compare it with RSNA bounding-box mask.

Formula:

```text
IoU = overlap area / union area
```

Higher IoU means the heatmap overlaps better with annotated pneumonia regions.

## 20. How Good Are the XAI Results?

For weakly supervised XAI on classification models, IoU values are usually modest.

Approximate interpretation:

| Mean IoU | Quality |
|---:|---|
| 0.05 to 0.15 | Weak localization |
| 0.15 to 0.25 | Moderate localization |
| 0.25 to 0.35 | Good for classification-only XAI |
| 0.35+ | Strong, often needs localization-aware training |

Your results:

- Grad-CAM: 0.2252, moderate.
- Grad-CAM++: 0.2825, good for classification-only XAI.
- Integrated Gradients: 0.1888, moderate.

Conclusion:

Grad-CAM++ performed best for localization in this project.

## 21. Generated Output Files

Important files:

```text
outputs/cxr_foundation_densenet121-res224-all/best_cxr_foundation.pt
outputs/feature_cache/densenet121-res224-all_train_full.npz
outputs/feature_cache/densenet121-res224-all_val_full.npz
outputs/threshold_tuning_densenet121-res224-all/threshold_tuning.json
outputs/xai_quantitative.json
```

Figures:

```text
outputs/figures/xai_comparison_pneumonia.png
outputs/figures/xai_comparison_normal.png
outputs/figures/xai_failure_cases.png
outputs/figures/metrics_table.png
outputs/figures/xai_quantitative.png
outputs/figures/confusion_matrix.png
outputs/figures/roc_curve.png
```

Also generated:

```text
outputs/figures/cases/
```

This contains 50 individual side-by-side XAI case overlays.

## 22. What to Say in an Interview

### 22.1 Short Project Explanation

I built a pneumonia classification and explainability pipeline using RSNA chest X-ray DICOM images. I used a TorchXRayVision DenseNet121 pretrained on chest X-ray datasets as a frozen feature extractor, cached 1024-dimensional features, trained a small MLP head with balanced focal loss, tuned the probability threshold for best F1, and implemented Grad-CAM, Grad-CAM++, and Integrated Gradients to explain predictions.

### 22.2 Why Use a Pretrained Medical Encoder?

Medical datasets are relatively specialized. A model pretrained on chest X-ray datasets learns radiological patterns better than a generic ImageNet model. Freezing the encoder also reduces compute requirements and overfitting risk.

### 22.3 Why Cache Features?

DenseNet feature extraction is expensive. Once we freeze the encoder, image features do not change during head training. So caching features lets us train and tune the MLP head quickly without recomputing DenseNet outputs every epoch.

### 22.4 Why Threshold Tuning?

The dataset is imbalanced, so the default 0.5 threshold was not optimal. It gave high recall but too many false positives. Tuning the threshold improved F1 from 0.4973 to 0.6433.

### 22.5 Why Use XAI?

In healthcare, predictions need to be interpretable. Heatmaps help check whether the model is looking at clinically meaningful lung regions rather than shortcuts or artifacts.

### 22.6 Which XAI Method Worked Best?

Grad-CAM++ had the highest mean IoU with bounding boxes:

```text
Grad-CAM++ IoU = 0.2825
```

It also remained fast compared to Integrated Gradients.

### 22.7 What Were the Main Challenges?

Main challenges:

- Removing abandoned TensorFlow dependencies from the PyTorch pipeline.
- Handling DICOM preprocessing correctly.
- Managing class imbalance.
- Avoiding Mac MPS instability during XAI.
- Making feature extraction efficient enough on a laptop.

### 22.8 What Would You Improve Next?

I would:

- Fine-tune DenseNet denseblock4.
- Run full 50-step Integrated Gradients on a more stable GPU environment.
- Calibrate probabilities.
- Try stronger augmentations.
- Add test-set evaluation after validation tuning.
- Compare against a fully fine-tuned CNN and possibly an ensemble.
- Evaluate fairness and robustness across patient subgroups if metadata is available.

## 23. Strengths of the Project

- End-to-end pipeline works locally.
- Uses medical-domain pretrained encoder.
- Handles DICOM data directly.
- Uses class imbalance-aware training.
- Uses threshold tuning instead of blindly using 0.5.
- Implements three explainability methods.
- Quantitatively evaluates XAI, not just visually.
- Produces all required figures and artifacts.

## 24. Limitations

- Encoder was frozen; full fine-tuning may improve results.
- Final XAI batch used `ig_steps=8` for safety, while the ideal Integrated Gradients setting is 50.
- Validation split was used for threshold tuning; test split should be used only once for final unbiased reporting.
- XAI IoU is limited because the classifier was not explicitly trained for localization.
- Bounding boxes are only available for pneumonia cases.
- AUC-ROC is strong, but precision/recall tradeoff still needs improvement for clinical deployment.

## 25. Final Result Summary

Classification:

```text
Best threshold: 0.731
Accuracy:       0.8294
Precision:      0.6081
Recall:         0.6829
F1:             0.6433
AUC-ROC:        0.8731
AUC-PR:         0.6809
```

XAI:

```text
Best localization method: Grad-CAM++
Grad-CAM++ Mean IoU:      0.2825
Grad-CAM++ Time:          537.5 ms
Consistency:              1.0000
```

Overall conclusion:

This is a strong project-level result. The classification AUC-ROC and F1 meet the target success criteria, and the explainability pipeline is complete with both qualitative figures and quantitative XAI comparison. For research-level performance, the next step would be partial DenseNet fine-tuning and more robust full-step XAI evaluation on a stable GPU environment.


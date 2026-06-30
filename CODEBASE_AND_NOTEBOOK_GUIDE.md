# Codebase and Notebook Guide

This document explains what each important file in this project does, how the files connect to each other, and what every major code block in the main notebook is meant to do.

The most important thing to remember is this:

**The final working project uses the DenseNet121 + PyTorch + TorchXRayVision pipeline.**

The repository also contains an older **ResNet50 + TensorFlow/Keras** path. That older path is useful for understanding the original project structure, but it was not the final path used for the completed training and XAI pipeline.

For interviews, say:

> The codebase originally had a ResNet50/TensorFlow implementation, but the final working pipeline was migrated to a frozen TorchXRayVision DenseNet121 encoder with a PyTorch MLP classification head. The final XAI methods were implemented on the DenseNet/PyTorch path.

---

## 1. Project Structure

At a high level, the project is organized like this:

```text
xai-pneumonia-cxr-main/
├── app.py
├── model_utils.py
├── CODEX_CHANGES.md
├── PROJECT_INTERVIEW_GUIDE.md
├── BEGINNER_CONCEPTS_AND_INTERVIEW_ANSWERS.md
├── CODEBASE_AND_NOTEBOOK_GUIDE.md
├── outputs/
├── rsna-pneumonia-detection-challenge/
├── scripts/
└── xai_pneumonia/
    ├── data/
    ├── model/
    ├── xai/
    ├── evaluation/
    ├── notebooks/
    ├── train_cxr_foundation.py
    ├── tune_threshold.py
    ├── train.py
    ├── evaluate.py
    └── requirements-mac-training.txt
```

The main folders are:

| Folder | Purpose |
|---|---|
| `xai_pneumonia/data/` | Dataset loading, metadata handling, train/val/test splits, DICOM image reading |
| `xai_pneumonia/model/` | Model definitions, including the final DenseNet121 TorchXRayVision model |
| `xai_pneumonia/xai/` | Explainable AI methods such as Grad-CAM, Grad-CAM++, and Integrated Gradients |
| `xai_pneumonia/evaluation/` | Evaluation utilities, metrics, plots, and older statistical analysis code |
| `xai_pneumonia/notebooks/` | Main notebook, mostly from the older ResNet/TensorFlow workflow |
| `outputs/` | Generated model checkpoints, cached features, metrics, figures, and XAI results |
| `rsna-pneumonia-detection-challenge/` | Local RSNA dataset folder |
| `scripts/` | Helper shell scripts for downloading or running training commands |

---

## 2. Final Pipeline Files

These are the most important files for the final completed DenseNet/PyTorch pipeline.

### `xai_pneumonia/train_cxr_foundation.py`

This is the **main final training script**.

It does the following:

1. Reads RSNA labels and image paths.
2. Loads DICOM chest X-rays.
3. Converts each X-ray into a tensor suitable for TorchXRayVision.
4. Uses a frozen pretrained DenseNet121 encoder.
5. Extracts 1024-dimensional image features.
6. Saves those features to `outputs/feature_cache/`.
7. Trains a small MLP classification head on top of the frozen features.
8. Uses focal loss to handle class imbalance.
9. Tracks validation metrics.
10. Saves the best checkpoint.

Important concepts inside this file:

| Concept | Meaning |
|---|---|
| `RSNATorchDataset` | PyTorch dataset class that reads RSNA images and labels |
| DICOM preprocessing | Converts medical images into normalized tensors |
| Feature caching | Saves DenseNet features so training the head becomes much faster |
| Frozen encoder | DenseNet weights are not trained; only the MLP head is trained |
| MLP head | Small neural network that turns DenseNet features into pneumonia probability |
| Focal loss | Loss function that helps when normal cases are more common than pneumonia cases |
| Early stopping | Stops training when validation F1 does not improve |

Interview explanation:

> This script is the core training pipeline. It treats TorchXRayVision DenseNet121 as a pretrained medical feature extractor and trains only a lightweight MLP classifier on top. To make training efficient on a MacBook, it caches the DenseNet features and reuses them across epochs.

---

### `xai_pneumonia/model/cxr_torch_model.py`

This file defines the **final DenseNet/PyTorch model components**.

It contains:

| Class | Purpose |
|---|---|
| `CXRFeatureExtractor` | Loads the TorchXRayVision DenseNet121 encoder and outputs 1024-dimensional features |
| `CXRMLPHead` | A small classifier head with hidden layer, dropout, and final output logit |
| `CXRClassifier` | Combines the feature extractor and MLP head into one model |

The final model idea:

```text
Chest X-ray
   ↓
TorchXRayVision DenseNet121 encoder
   ↓
1024-dimensional feature vector
   ↓
MLP head
   ↓
Pneumonia probability
```

Why this is useful:

1. DenseNet121 is already trained on many chest X-ray tasks.
2. The model learns medical visual features better than a generic ImageNet model.
3. Freezing the encoder reduces training time and memory usage.
4. Training only the MLP head is practical on a MacBook with MPS.

Interview explanation:

> The model file separates feature extraction from classification. The DenseNet encoder learns high-level chest X-ray features, and the MLP head learns the final binary decision: pneumonia or normal.

---

### `xai_pneumonia/data/data_loader.py`

This file handles **dataset loading and split management**.

It does the following:

1. Reads `stage_2_train_labels.csv`.
2. Locates DICOM files in `stage_2_train_images/`.
3. Builds metadata for each patient/image.
4. Handles pneumonia labels.
5. Handles bounding box annotations for pneumonia cases.
6. Loads or creates train/validation/test splits.
7. Computes class distribution and class weights.

Important data facts:

| Item | Meaning |
|---|---|
| `patientId` | Unique ID for each chest X-ray |
| `Target` | `1` means pneumonia, `0` means no pneumonia |
| Bounding boxes | Available only for some positive pneumonia cases |
| `splits.json` | Stores fixed train/val/test split IDs |

Interview explanation:

> This file prepares the RSNA dataset for the rest of the pipeline. It reads labels, links each label to its DICOM image, preserves bounding boxes for localization evaluation, and provides consistent train/validation/test splits.

---

### `xai_pneumonia/tune_threshold.py`

This script finds the **best classification threshold**.

Normally, a binary classifier uses threshold `0.5`:

```text
probability >= 0.5 → pneumonia
probability < 0.5  → normal
```

But because the dataset is imbalanced, `0.5` may not give the best F1 score.

This script:

1. Loads the trained checkpoint.
2. Loads cached validation features.
3. Computes probabilities for validation images.
4. Tests many possible thresholds.
5. Finds the threshold that gives the best F1 score.
6. Prints metrics such as accuracy, precision, recall, F1, AUC-ROC, AUC-PR, and confusion matrix.

Interview explanation:

> Threshold tuning converts model probabilities into final class labels in a way that optimizes the validation F1 score. This is important because the dataset is imbalanced and the best decision threshold is not always 0.5.

---

### `xai_pneumonia/xai/explain.py`

This is the **main final XAI script**.

It implements and runs:

1. Grad-CAM
2. Grad-CAM++
3. Integrated Gradients

For each selected image, it can:

1. Load the raw image.
2. Run the trained DenseNet/PyTorch classifier.
3. Generate heatmaps from all three XAI methods.
4. Overlay heatmaps on the original X-ray.
5. Save comparison figures.
6. Compute quantitative XAI metrics.

Important outputs:

| Output | Meaning |
|---|---|
| `xai_comparison_pneumonia.png` | Pneumonia examples with all XAI methods |
| `xai_comparison_normal.png` | Normal examples with all XAI methods |
| `xai_failure_cases.png` | Misclassified examples with explanations |
| `xai_quantitative.png` | XAI comparison chart |
| XAI CSV/JSON files | Quantitative results such as IoU, time, and consistency |

Interview explanation:

> This file explains the model's predictions. Grad-CAM and Grad-CAM++ use gradients from the final convolutional layer, while Integrated Gradients attributes the prediction back to input pixels by comparing the image to a black baseline.

---

### `xai_pneumonia/explain.py`

This is a **compatibility wrapper**.

It exists so older commands that call:

```bash
python -m xai_pneumonia.explain
```

can still route into the newer XAI implementation.

The actual final implementation lives in:

```text
xai_pneumonia/xai/explain.py
```

Interview explanation:

> This wrapper keeps older entry points working while the real DenseNet/PyTorch XAI logic lives in the newer XAI module.

---

### `xai_pneumonia/requirements-mac-training.txt`

This file lists the dependencies needed for training and XAI on Mac.

Important packages:

| Package | Why it is used |
|---|---|
| `torch` | PyTorch deep learning framework |
| `torchvision` | Image utilities and model support |
| `torchxrayvision` | Pretrained medical chest X-ray models |
| `captum` | Integrated Gradients implementation |
| `grad-cam` | Grad-CAM++ implementation |
| `pydicom` | Reading DICOM medical image files |
| `opencv-python` | Image processing and heatmap overlays |
| `scikit-learn` | Metrics such as F1, AUC, confusion matrix |
| `matplotlib` | Saving plots and figures |
| `tqdm` | Progress bars |

Interview explanation:

> The requirements file was adjusted for a Mac training environment and includes PyTorch, TorchXRayVision, Captum, Grad-CAM, DICOM support, plotting, and evaluation libraries.

---

## 3. Legacy or Supporting Files

These files are present in the repository but are not the final DenseNet/PyTorch path.

### `xai_pneumonia/train.py`

This is the older training script for the TensorFlow/ResNet50 path.

It is not the final script used for the completed pipeline.

Interview explanation:

> This belongs to the older ResNet50 implementation. The final project moved away from it and used `train_cxr_foundation.py` instead.

---

### `xai_pneumonia/model/resnet50_model.py`

This defines the older ResNet50 model architecture.

It is useful for understanding the original design but should not be described as the final model.

Final model file:

```text
xai_pneumonia/model/cxr_torch_model.py
```

---

### `xai_pneumonia/evaluate.py`

This is mostly from the older evaluation flow.

The final DenseNet/PyTorch pipeline uses:

1. Metrics printed by `train_cxr_foundation.py`
2. Threshold tuning from `tune_threshold.py`
3. XAI quantitative outputs from `xai/xai/explain.py`

---

### `xai_pneumonia/evaluation/metrics.py`

This contains reusable metric functions.

It may include functions for:

1. Accuracy
2. Precision
3. Recall
4. F1 score
5. AUC-ROC
6. AUC-PR
7. Confusion matrix
8. XAI localization metrics

Interview explanation:

> This file centralizes evaluation logic so metrics can be reused by training, threshold tuning, and analysis scripts.

---

### `xai_pneumonia/evaluation/visualizer.py`

This contains helper functions for plotting evaluation results and XAI visualizations.

It belongs more to the older notebook-style workflow.

---

### `xai_pneumonia/evaluation/statistical_tests.py`

This file contains statistical testing utilities, such as comparing XAI methods or classification results.

Possible examples:

1. Wilcoxon signed-rank test
2. McNemar test
3. Statistical comparison of method outputs

Interview explanation:

> This file is used when we want to go beyond visual comparison and statistically compare whether two methods behave differently.

---

### `xai_pneumonia/xai/gradcam.py`

Older standalone Grad-CAM implementation.

The final DenseNet/PyTorch Grad-CAM implementation is inside:

```text
xai_pneumonia/xai/explain.py
```

---

### `xai_pneumonia/xai/gradcam_plus_plus.py`

Older standalone Grad-CAM++ implementation.

The final implementation uses the DenseNet/PyTorch path and may use the `pytorch-grad-cam` package where appropriate.

---

### `xai_pneumonia/xai/integrated_gradients.py`

Older standalone Integrated Gradients implementation.

The final implementation uses Captum through:

```text
xai_pneumonia/xai/explain.py
```

---

### `xai_pneumonia/xai/common.py`

Shared helper functions for XAI image loading, preprocessing, heatmap normalization, and overlays.

This type of file prevents repeated code across multiple XAI methods.

---

### `xai_pneumonia/xai/layer_selector.py`

Helper logic for selecting model layers for Grad-CAM.

In the final DenseNet pipeline, the target Grad-CAM layer is:

```text
features.denseblock4
```

Interview explanation:

> Grad-CAM needs a convolutional feature map. For DenseNet121, the final dense block is a good target because it contains high-level spatial features.

---

### `xai_pneumonia/utils.py`

General helper functions used across the project.

Typical utility files contain reusable code for:

1. Seeding
2. Directory creation
3. JSON reading/writing
4. Logging
5. Common formatting

---

## 4. Script Files

### `scripts/download_rsna_train_only.sh`

Helper script for downloading only the RSNA training data.

For this project run, the dataset was already present locally, so downloading was not needed.

---

### `scripts/train_cxr_foundation_min_storage.sh`

Helper shell script for running the DenseNet/TorchXRayVision training pipeline in a storage-conscious way.

It is related to the final path.

---

### `scripts/train_resnet_min_storage.sh`

Helper shell script for the older ResNet50 path.

It is not part of the final DenseNet/PyTorch result.

---

## 5. Streamlit App Files

The user specifically asked to keep these untouched for now.

### `app.py`

Streamlit app entry point.

It likely handles:

1. Uploading an image.
2. Loading a model.
3. Showing prediction results.
4. Displaying explanation images.

This file was not modified during the final pipeline work.

---

### `model_utils.py`

Utility functions used by the Streamlit app.

It may contain model loading, preprocessing, or prediction helper functions for the app.

This file was also kept untouched.

---

## 6. Output Files and Folders

### `outputs/feature_cache/`

Stores cached DenseNet feature vectors.

Feature caching is important because DenseNet feature extraction is expensive.

Instead of recomputing features every epoch:

```text
X-ray → DenseNet → 1024 features
```

the features are saved once and reused:

```text
cached 1024 features → MLP training
```

This makes MLP training much faster.

---

### `outputs/cxr_foundation_densenet121-res224-all/`

Stores the final model outputs for the DenseNet121 run.

Important file:

```text
best_cxr_foundation.pt
```

This is the best saved PyTorch checkpoint.

---

### `outputs/figures/`

Stores generated figures, such as:

1. Confusion matrix
2. ROC curve
3. Metrics table
4. XAI pneumonia comparison
5. XAI normal comparison
6. XAI failure cases
7. XAI quantitative comparison

---

## 7. How the Final Pipeline Connects

The final pipeline flow is:

```text
RSNA DICOM images + labels
        ↓
xai_pneumonia/data/data_loader.py
        ↓
xai_pneumonia/train_cxr_foundation.py
        ↓
xai_pneumonia/model/cxr_torch_model.py
        ↓
DenseNet121 feature extraction
        ↓
outputs/feature_cache/
        ↓
MLP classifier training
        ↓
best_cxr_foundation.pt
        ↓
xai_pneumonia/tune_threshold.py
        ↓
best classification threshold and metrics
        ↓
xai_pneumonia/xai/explain.py
        ↓
Grad-CAM, Grad-CAM++, Integrated Gradients
        ↓
outputs/figures/
```

Interview explanation:

> The dataset loader prepares images and labels. The training script uses the model file to extract DenseNet features and train the MLP head. The threshold tuning script converts probabilities into better final labels. The XAI script then explains predictions using three methods and saves both qualitative and quantitative comparisons.

---

## 8. Main Notebook

The main notebook is:

```text
xai_pneumonia/notebooks/full_pipeline.ipynb
```

Important note:

**This notebook mainly documents the older TensorFlow/ResNet50 workflow.**

Some cells are still useful for explaining the overall project idea, EDA, dataset inspection, split verification, visualizations, and XAI concepts. However, the final actual execution used the DenseNet/PyTorch scripts.

For interviews, say:

> The notebook was originally built around the ResNet50/TensorFlow version of the project. For the final implementation, the operational pipeline was moved into PyTorch scripts using TorchXRayVision DenseNet121. So the notebook is useful as a walkthrough, but the final reproducible commands are script-based.

---

## 9. Notebook Code Block Walkthrough

The notebook has 36 cells. Below is what each cell does.

### Cell 1: Markdown Title and Overview

Introduces the project as an explainable AI pipeline for pneumonia detection from chest X-rays.

It describes the general workflow:

1. Load dataset.
2. Train classifier.
3. Generate XAI heatmaps.
4. Evaluate classification and explainability.

Final pipeline relevance:

Useful for project explanation, but not specific to the final DenseNet implementation.

---

### Cell 2: Imports, Paths, and Global Settings

This code cell imports common libraries:

1. `os`
2. `sys`
3. `json`
4. `random`
5. `subprocess`
6. `pathlib`
7. `matplotlib`
8. `numpy`
9. `pandas`
10. `seaborn`
11. `tensorflow`

It also:

1. Sets random seeds.
2. Defines project root paths.
3. Defines dataset and output paths.
4. Adds the project root to `sys.path`.
5. Sets notebook plotting style.

Final pipeline relevance:

Partly useful, but it imports TensorFlow because the notebook belongs to the older ResNet path. The final pipeline uses PyTorch instead.

Interview explanation:

> This cell prepares the notebook environment by importing libraries, fixing random seeds for reproducibility, and defining where data and outputs are stored.

---

### Cell 3: Markdown Section for Environment Check

Introduces the environment verification section.

---

### Cell 4: Environment and Dependency Check

This code checks installed package versions.

It imports and prints versions for:

1. Python
2. TensorFlow
3. NumPy
4. Pandas
5. OpenCV
6. pydicom
7. scikit-learn
8. SciPy
9. scikit-image

It also checks whether TensorFlow can see a GPU.

Final pipeline relevance:

Conceptually useful, but the final project should check PyTorch MPS instead:

```python
import torch
print(torch.backends.mps.is_available())
```

Interview explanation:

> This block verifies that the environment has the required libraries. In the final Mac implementation, the important backend check is PyTorch MPS availability rather than TensorFlow GPU availability.

---

### Cell 5: Markdown Section for Dataset EDA

Introduces exploratory data analysis.

EDA means looking at the data before training.

---

### Cell 6: Load Metadata and Plot Class Distribution

This cell imports `RSNAPneumoniaDataModule`, creates a data module, and loads metadata.

It then:

1. Displays the first few rows of the metadata table.
2. Counts normal and pneumonia samples.
3. Plots the class distribution.

Final pipeline relevance:

Useful. Even though the final training script is different, the same dataset ideas apply.

Interview explanation:

> This block helps us understand the dataset. It shows how many normal and pneumonia images we have and confirms class imbalance.

---

### Cell 7: Display Random Training Images

This cell samples 16 training images and displays them in a grid.

Purpose:

1. Confirm DICOM images can be read.
2. Visually inspect image quality.
3. Check that labels look reasonable.

Final pipeline relevance:

Useful for data sanity checking.

Interview explanation:

> Before training, we visually inspect random chest X-rays to make sure the loader is reading images correctly and the dataset looks valid.

---

### Cell 8: Display Pneumonia Images with Bounding Boxes

This cell selects pneumonia-positive cases that have bounding boxes.

It:

1. Reads the DICOM image.
2. Converts it to a displayable image.
3. Draws bounding boxes around pneumonia regions.
4. Displays example annotated images.

Final pipeline relevance:

Very useful for understanding XAI localization evaluation.

Interview explanation:

> RSNA provides bounding boxes for some pneumonia cases. We use these boxes to compare whether the XAI heatmap focuses near the annotated pneumonia region.

---

### Cell 9: Markdown Section for Split Verification

Introduces train/validation/test split checking.

---

### Cell 10: Verify Train/Validation/Test Splits

This cell prints and plots split statistics.

It checks:

1. Number of samples in train split.
2. Number of samples in validation split.
3. Number of samples in test split.
4. Pneumonia ratio in each split.
5. Whether `splits.json` can be loaded.

Final pipeline relevance:

Useful and important.

Expected split sizes are approximately:

| Split | Approximate size |
|---|---:|
| Train | 18,678 |
| Validation | 4,003 |
| Test | 4,003 |

Interview explanation:

> We verify the splits to make sure training, validation, and testing are separate and have similar class distributions. This avoids data leakage and makes evaluation more trustworthy.

---

### Cell 11: Markdown Section for Preprocessing

Introduces image preprocessing.

---

### Cell 12: Show Preprocessing and Augmentation

This cell loads one sample and displays:

1. The resized RGB image.
2. A standardized channel.
3. An augmented version.

Purpose:

1. Show how raw medical images are transformed before model input.
2. Confirm preprocessing does not break the image.
3. Demonstrate augmentation.

Final pipeline relevance:

Conceptually useful, but exact preprocessing differs between old TensorFlow path and final PyTorch/TorchXRayVision path.

Interview explanation:

> Preprocessing converts DICOM images into model-ready tensors. For the final DenseNet model, images are resized and normalized in the format expected by TorchXRayVision.

---

### Cell 13: Markdown Section for Model Architecture

Introduces the model architecture section.

---

### Cell 14: Build and Display ResNet50 Model

This cell imports and builds the old ResNet50 model.

It prints:

1. Model summary.
2. Architecture details.

Final pipeline relevance:

Legacy only.

The final model is not ResNet50. The final model is:

```text
TorchXRayVision DenseNet121 encoder + MLP head
```

Interview explanation:

> This notebook cell shows the older ResNet50 architecture. In the final implementation, we replaced this path with TorchXRayVision DenseNet121 because it is pretrained on chest X-ray data and better suited to the domain.

---

### Cell 15: Markdown Section for Phase 1 Training

Introduces the first training phase in the old notebook.

---

### Cell 16: Run Old Training Script

This cell builds a command for:

```bash
python train.py
```

It passes arguments such as:

1. Data directory.
2. Output directory.
3. Number of epochs.
4. Batch size.

Then it runs the command using `subprocess.run`.

Final pipeline relevance:

Legacy only.

The final training script is:

```bash
python -m xai_pneumonia.train_cxr_foundation
```

Interview explanation:

> The notebook originally launched TensorFlow/ResNet training from this cell. In the final version, training is run through the DenseNet/PyTorch script instead.

---

### Cell 17: Markdown Section for Phase 2 Fine-Tuning

Introduces fine-tuning in the old workflow.

Final pipeline relevance:

Legacy concept.

In the final project, the DenseNet encoder is frozen and the MLP head is trained.

---

### Cell 18: Load Training History

This cell reads:

```text
model/history.json
```

and displays the training history as a DataFrame.

It is used to inspect metrics across epochs.

Final pipeline relevance:

Conceptually useful, but final metrics are produced by the PyTorch training and threshold tuning scripts.

Interview explanation:

> Training history lets us inspect how loss and metrics changed over epochs and whether the model improved or overfit.

---

### Cell 19: Markdown Section for Training Curves

Introduces training curve visualization.

---

### Cell 20: Display Training Curves

This cell opens:

```text
outputs/figures/training_curves.png
```

and displays it in the notebook.

Training curves usually show:

1. Training loss.
2. Validation loss.
3. Training metric.
4. Validation metric.

Final pipeline relevance:

Useful conceptually.

Interview explanation:

> Training curves help diagnose underfitting, overfitting, and whether training is stable.

---

### Cell 21: Markdown Section for Grad-CAM Layer Selection

Introduces selection of the convolutional layer used for Grad-CAM.

Grad-CAM needs a convolutional layer because it produces a spatial feature map.

---

### Cell 22: Compare Candidate Grad-CAM Layers

This cell loads the older trained ResNet model and compares Grad-CAM heatmaps from different layers:

1. `conv3_block4_out`
2. `conv4_block6_out`
3. `conv5_block3_out`

It plots heatmaps and chooses the best-looking layer.

Final pipeline relevance:

Legacy implementation, but the idea is still important.

For the final DenseNet model, the target layer is:

```text
features.denseblock4
```

Interview explanation:

> Grad-CAM layer choice matters because shallow layers may be too low-level and very deep layers may be too coarse. For DenseNet121, the final dense block gives high-level spatial features suitable for heatmaps.

---

### Cell 23: Markdown Section for XAI Method Explanations

Introduces the explainability methods:

1. Grad-CAM
2. Grad-CAM++
3. Integrated Gradients

---

### Cell 24: Run Batch XAI Generation

This cell calls an explanation script using `subprocess.run`.

It attempts to generate XAI outputs for multiple examples.

Final pipeline relevance:

Partly outdated.

The final XAI command should use:

```bash
python -m xai_pneumonia.xai.explain
```

and the correct DenseNet checkpoint arguments.

Interview explanation:

> This cell was designed to run batch XAI generation from the notebook. The final project runs XAI through the updated DenseNet/PyTorch XAI script.

---

### Cell 25: Markdown Section for Integrated Gradients Baseline Ablation

Introduces comparison of different Integrated Gradients baselines.

In Integrated Gradients, the baseline is the reference image.

Examples:

1. Black image baseline
2. White image baseline
3. Blurred image baseline

Final pipeline relevance:

Conceptually useful.

The final implementation uses a zero tensor black baseline.

---

### Cell 26: Display IG Baseline Comparison Figure

This cell looks for files named like:

```text
ig_baseline_comparison_*.png
```

and displays the first one found.

Final pipeline relevance:

Mostly legacy, but helpful for explaining what baseline ablation means.

Interview explanation:

> Baseline ablation checks whether Integrated Gradients explanations change significantly depending on the chosen reference image.

---

### Cell 27: Markdown Section for Aggregate XAI Metrics

Introduces quantitative XAI evaluation.

---

### Cell 28: Run Full XAI Evaluation

This cell runs explanation and evaluation scripts over the test set.

It is intended to produce aggregate XAI metrics.

Final pipeline relevance:

Mostly legacy command structure.

The final DenseNet XAI evaluation is handled by:

```bash
python -m xai_pneumonia.xai.explain
```

Interview explanation:

> This block represents the idea of evaluating XAI methods quantitatively instead of only looking at heatmaps visually.

---

### Cell 29: Markdown Section for Statistical Significance

Introduces statistical testing of results.

---

### Cell 30: Display Statistical Test Results

This cell reads:

```text
evaluation_summary.json
```

and displays:

1. Wilcoxon test results for XAI metrics.
2. McNemar test results for classification comparison.

Final pipeline relevance:

Conceptually useful, but not central to the final DenseNet pipeline outputs.

Interview explanation:

> Statistical tests help determine whether observed differences between methods are likely meaningful or just random variation.

---

### Cell 31: Markdown Section for Classification Metrics

Introduces final classification result visualization.

---

### Cell 32: Display Confusion Matrix and ROC/PR Curves

This cell displays:

1. Confusion matrix figure.
2. ROC and precision-recall curve figure.

Final pipeline relevance:

Useful.

The final project generated similar figures in:

```text
outputs/figures/
```

Interview explanation:

> The confusion matrix shows true positives, false positives, true negatives, and false negatives. ROC and PR curves show model performance across different thresholds.

---

### Cell 33: Markdown Section for Failure Mode Analysis

Introduces analysis of misclassified examples.

Failure mode analysis means studying where the model makes mistakes.

---

### Cell 34: Display XAI Failure or Comparison Images

This cell searches for files like:

```text
xai_comparison_*.png
```

and displays up to six of them.

Final pipeline relevance:

Useful.

The final project generated comparison figures for:

1. Correct pneumonia cases.
2. Correct normal cases.
3. Misclassified cases.

Interview explanation:

> Failure case analysis helps us understand whether mistakes happen because the image is difficult, the model focuses on the wrong region, or the class boundary is ambiguous.

---

### Cell 35: Markdown Section for Conclusions

Introduces final conclusions and method recommendation.

---

### Cell 36: Select Best XAI Method from Summary

This cell reads the XAI comparison table from the evaluation summary.

It identifies:

1. Best method by Mean IoU.
2. Fastest method by runtime.
3. Most stable method by SSIM or consistency.

Then it prints a recommendation.

Final pipeline relevance:

Conceptually useful.

The final project compared methods using:

1. Mean IoU
2. Computation time
3. Consistency

Interview explanation:

> This block summarizes the XAI comparison. A good explanation method should localize the disease region well, run efficiently, and produce stable results across repeated runs.

---

## 10. Final Commands to Know

These are the kinds of commands used in the final DenseNet/PyTorch workflow.

### Check MPS

```bash
.venv-train/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Meaning:

Checks whether PyTorch can use Apple Silicon GPU acceleration through MPS.

---

### Train DenseNet Foundation Model

```bash
.venv-train/bin/python -m xai_pneumonia.train_cxr_foundation
```

Meaning:

Runs final DenseNet feature extraction, feature caching, and MLP head training.

---

### Tune Threshold

```bash
.venv-train/bin/python -m xai_pneumonia.tune_threshold
```

Meaning:

Finds the best probability threshold on validation data.

---

### Run XAI

```bash
.venv-train/bin/python -m xai_pneumonia.xai.explain
```

Meaning:

Generates Grad-CAM, Grad-CAM++, Integrated Gradients overlays and quantitative XAI comparison results.

---

## 11. What to Say in an Interview

If asked what the project does:

> This project classifies chest X-rays as pneumonia or normal using the RSNA Pneumonia Detection dataset. The final model uses a frozen TorchXRayVision DenseNet121 encoder to extract medical image features and a trained MLP head for binary classification. After training, I used threshold tuning to optimize F1 score and implemented three explainability methods: Grad-CAM, Grad-CAM++, and Integrated Gradients.

If asked why DenseNet121:

> DenseNet121 from TorchXRayVision is pretrained on chest X-ray datasets, so it already understands medical imaging patterns better than a generic natural-image model. Freezing it also makes training faster and more stable on a MacBook.

If asked why not ResNet50:

> The repository had an older ResNet50/TensorFlow path, but the final project requirement was to use the DenseNet/PyTorch path. DenseNet with TorchXRayVision was more appropriate because it is pretrained for chest X-rays.

If asked what each major file does:

> `data_loader.py` prepares the dataset, `cxr_torch_model.py` defines the DenseNet feature extractor and MLP classifier, `train_cxr_foundation.py` trains the classifier and caches features, `tune_threshold.py` finds the best decision threshold, and `xai/explain.py` generates and evaluates explanations.

If asked what the notebook does:

> The notebook is a walkthrough of the original project pipeline. It covers environment checks, dataset exploration, split verification, preprocessing visualization, model training, XAI generation, and evaluation. However, many notebook cells refer to the older ResNet/TensorFlow path, while the final working pipeline is script-based and uses DenseNet/PyTorch.

---

## 12. Simple Mental Model

Remember the project in five stages:

```text
1. Data
   Load RSNA DICOM X-rays and labels.

2. Model
   Use pretrained DenseNet121 to extract medical image features.

3. Training
   Train an MLP head to classify pneumonia vs normal.

4. Evaluation
   Tune threshold and compute classification metrics.

5. Explainability
   Use Grad-CAM, Grad-CAM++, and Integrated Gradients to show where the model looked.
```

That is the cleanest way to explain the full project.


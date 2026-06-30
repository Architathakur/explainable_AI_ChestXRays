# CODEX Changes

## Streamlit PneumoXAI app

Created these standalone app files at the project root, separate from the existing notebooks and `xai_pneumonia/` research package:

- `app.py`
- `model_utils.py`
- `requirements.txt`
- `packages.txt`
- `README_SPACES.md`
- `.streamlit/config.toml`

Implementation decisions:

- Kept all Streamlit app logic in root-level files so the notebook pipeline remains clean and unchanged.
- Did not modify anything under `xai_pneumonia/`.
- Used `Path(__file__).parent` for all model, threshold, figure, and TorchXRayVision cache paths.
- Implemented CPU-only model loading, inference, and XAI with no CUDA, MPS, or device auto-detection.
- Mirrored the trained architecture: TorchXRayVision DenseNet121 encoder plus MLP head `Linear(1024,512) -> ReLU -> Dropout(0.4) -> Linear(512,2)`.
- Loaded `head_state_dict` from the checkpoint and optionally loaded `encoder_state_dict` if the checkpoint includes fine-tuned encoder weights.
- Added checkpoint compatibility for the existing saved MLP head keys (`net.0.*`, `net.3.*`) while keeping the standalone app head as the requested `nn.Sequential`.
- Loaded the tuned threshold from `best_threshold` when present, otherwise from the existing `best_f1.threshold` key, falling back to `0.731`.
- Added shape assertions for model input, logits, Grad-CAM activations, and Integrated Gradients attributions with clear error messages.
- Implemented DICOM preprocessing with MONOCHROME1 inversion and a PIL grayscale fallback with a Streamlit warning.
- Implemented Grad-CAM manually against `model.encoder.features.denseblock4`.
- Implemented Grad-CAM++ using `pytorch_grad_cam` when available, with a manual Grad-CAM++ fallback.
- Implemented Integrated Gradients with Captum using a zero baseline and `n_steps=20`.
- Logged XAI timings to console with `print()` for all three XAI methods.
- Built the requested dark Streamlit UI with sidebar logo/toggles, model metadata, legend, uploader styling, prediction cards, XAI grid, metric cards, confusion matrix, XAI comparison table, and figure placeholders.
- Added HuggingFace Spaces dependency files and Spaces card metadata.
- Replaced deprecated `st.image(..., use_column_width=True)` calls with `use_container_width=True` so Streamlit warning boxes do not appear in the app UI.
- Softened the research-use advisory styling so it reads as a restrained notice instead of a heavy yellow warning block.
- Added a plain-English explainability guide above the XAI maps explaining warm/cool colors, consistency across maps, and the clinical limitation.
- Rewrote XAI method captions in non-technical language while preserving method names and a short context note for each map.
- Added compact technical notes under each XAI caption explaining gradients, higher-order weighting, Integrated Gradients baseline/path attribution, and IoU scores for credibility.

- Made TensorFlow imports lazy in shared utilities and data loading so the DenseNet/PyTorch path can run in a Mac MPS environment without TensorFlow installed.
- Removed the package-level TensorFlow import that blocked all PyTorch-only module imports.
- Removed TensorFlow imports from subpackage initializers so PyTorch modules can be imported without the abandoned TensorFlow stack installed.
- Fixed `get_project_root()` to resolve to the project root, while preserving packaged split metadata under `xai_pneumonia/data`.
- Updated Mac training requirements with PyTorch, TorchXRayVision, Captum, and `grad-cam`.
- Updated DenseNet training defaults to `densenet121-res224-all`, project-level `outputs/`, hidden dim `512`, dropout `0.4`, balanced focal loss, and early stopping on validation F1.
- Expanded validation reports to include precision, recall, AUC-ROC, AUC-PR, and confusion matrix.
- Updated threshold tuning defaults to match the requested MLP head and print a final metrics table.
- Replaced the legacy TensorFlow explanation entrypoint with a DenseNet/PyTorch wrapper and added `xai_pneumonia/xai/explain.py` implementing Grad-CAM, Grad-CAM++, Integrated Gradients, case grids, metrics figures, and quantitative XAI evaluation.
- Redirected TorchXRayVision pretrained-weight caching to project-local `outputs/torchxrayvision` to avoid writing outside the workspace.
- Fixed Grad-CAM/Grad-CAM++ generation for a frozen encoder by enabling gradients on the explanation input tensor.
- Added an `--ig_steps` option to the DenseNet XAI CLI so Integrated Gradients can be smoke-tested with fewer steps before running the full 50-step evaluation.
- Added `--num_workers` and tqdm progress bars to DenseNet feature extraction so full DICOM caching is observable and can use multiple loader workers.
- Expanded generated `metrics_table.png` to include F1 and AUC-ROC along with accuracy, precision, recall, and AUC-PR.

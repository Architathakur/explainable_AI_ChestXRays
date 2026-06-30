---
title: PneumoXAI
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
tags:
  - medical-imaging
  - explainability
  - chest-xray
  - pneumonia
  - grad-cam
  - deep-learning
---

PneumoXAI — Pneumonia detection from chest X-rays using a TorchXRayVision DenseNet121 encoder (densenet121-res224-all) with Grad-CAM, Grad-CAM++, and Integrated Gradients explainability. Upload a frontal chest X-ray (DICOM or PNG/JPG) to get a binary pneumonia/normal prediction with XAI heatmap overlays. Trained on the RSNA Pneumonia Detection Challenge dataset. AUC-ROC: 0.873. Research use only — not a clinical diagnostic tool.

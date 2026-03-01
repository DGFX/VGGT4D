---
title: VGGT4D
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
python_version: "3.10"
preload_from_hub:
  - facebook/VGGT_tracker_fixed model_tracker_fixed_e20.pt
suggested_hardware: zero-a10g
startup_duration_timeout: 30m
---

# VGGT4D — 4D Scene Understanding from Monocular Video

Upload a short video to predict **depth maps**, **camera poses**, and **dynamic/static segmentation masks**.

The pipeline runs three stages:
1. Predict depth and initial dynamic map from attention patterns
2. Re-estimate camera poses masking dynamic regions
3. Refine dynamic masks via geometric consistency

# VGGT4D

4D scene understanding from monocular video. Extends [VGGT](https://github.com/facebookresearch/vggt) (Visual Geometry Grounded Transformer) with dynamic object segmentation — predicts per-frame depth, camera poses, and dynamic/static masks from a sequence of images.

## Setup

```bash
pip install torch torchvision  # PyTorch with CUDA
pip install -r requirements.txt
```

Checkpoint: place at `./ckpts/model_tracker_fixed_e20.pt` (auto-downloaded from `facebook/VGGT_tracker_fixed` if missing)

## Key Entry Points

**HF Space / Gradio app** — ZeroGPU-enabled web demo:
```bash
python app.py
```
Accepts video upload, runs 3-stage pipeline, returns depth video + dynamic mask overlay video + gallery. Uses `@spaces.GPU(duration=180)` for ZeroGPU. Model loaded at module level with `.cuda()` (ZeroGPU intercepts). Env vars: `CKPT_REPO`, `CKPT_FILENAME`, `CKPT_LOCAL`.


**Demo** — run full 3-stage pipeline on image directories:
```bash
python demo_vggt4d.py --input_dir <dir_of_scene_dirs> --output_dir <output>
```
Input: directory containing scene subdirectories, each with .jpg/.png images. Outputs: depth maps, camera poses (TUM format), intrinsics, dynamic masks.

**Visualization** — interactive 3D viewer (viser on port 8080):
```bash
python vis_vggt4d.py <scene_output_dir>
```

**Evaluation** — dynamic mask evaluation against DAVIS:
```bash
python eval_mask.py
```
Expects predictions at `outputs/ours/` and DAVIS GT at `datasets/DAVIS/Annotations_unsupervised/480p/`.

**Training** — DDP training with Hydra configs:
```bash
cd training && python launch.py --config default
```
Config at `training/config/default.yaml`. Requires CO3D dataset paths. Uses `torchrun` for multi-GPU.

## Architecture

### Module Layout

- `vggt/` — Base VGGT model (from Meta). ViT-based encoder with alternating attention.
  - `models/vggt.py` — `VGGT` model class, `models/aggregator.py` — `Aggregator` (ViT backbone)
  - `layers/` — attention, block, patch_embed, RoPE, SwiGLU FFN
  - `heads/` — `CameraHead` (pose encoding), `DPTHead` (depth & point maps), `TrackHead`
  - `utils/` — `pose_enc.py` (pose encoding conversion), `geometry.py`, `rotation.py`, `load_fn.py`

- `vggt4d/` — 4D extension. Inherits from vggt, adds dynamic mask support.
  - `models/vggt4d.py` — `VGGTFor4D` (extends `VGGT`), uses `AggregatorFor4D`
  - `models/aggregator.py` — `AggregatorFor4D` (extends `Aggregator`), captures q/k attention maps, accepts `dyn_masks`
  - `layers/attention.py` — `AttentionFor4D` adds `attention_with_dynamic_mask` (masks dynamic regions in attention)
  - `layers/block.py` — `BlockFor4D` wires dynamic masks through attention
  - `masks/dynamic_mask.py` — extract dynamic maps from attention q/k, clustering, multi-Otsu thresholding
  - `masks/refine_dyn_mask.py` — `RefineDynMask` refines masks via depth reprojection consistency
  - `visualization/scene.py` — `Scene4D` loads outputs for viser visualization

- `training/` — Training infrastructure
  - `trainer.py` — `Trainer` class (DDP), `launch.py` — entry point
  - `loss.py` — `MultitaskLoss` (camera, depth, point, track losses)
  - `data/` — datasets (CO3D, Virtual KITTI), dynamic dataloader

### 3-Stage Inference Pipeline (demo_vggt4d.py)

1. **Stage 1** — Run model on raw images. Get depth, pose, world points. Extract dynamic maps from attention q/k patterns (clustering + multi-Otsu threshold).
2. **Stage 2** — Re-run model with dynamic masks applied. Attention masks out dynamic regions, giving refined camera extrinsics for static background.
3. **Stage 3** — Refine dynamic masks via geometric consistency (depth reprojection between frames using refined poses).

### Key Conventions

- **Pose encoding**: 9-dim vector `[T(3), quat(4), FoV(2)]` — absolute translation, quaternion rotation, field of view. Converted to/from 3x4 extrinsic + 3x3 intrinsic via `vggt/utils/pose_enc.py`.
- **Coordinate system**: OpenCV convention (x-right, y-down, z-forward). Extrinsics are world-to-camera `[R|t]`.
- **Image preprocessing**: resize to 518px long edge, dimensions divisible by 14 (patch size). Values in [0,1].
- **Alternating attention**: each aggregator block runs frame attention (within-frame, tokens shape `B*S, P, C`) then global attention (across-frame, tokens shape `B, S*P, C`). 24 blocks total.
- **Token layout**: per frame = `[camera_token(1), register_tokens(4), patch_tokens(H/14 * W/14)]`. `patch_start_idx = 5`.
- **Tensor naming**: `B` = batch, `S` = sequence/num frames, `H/W` = spatial, `P` = total tokens per frame, `C` = embed dim (1024). Heads receive concatenated frame+global features (dim 2048).
- **Output dict keys**: `pose_enc`, `depth`, `depth_conf`, `world_points`, `world_points_conf`, `extrinsic`, `intrinsic`, `cam2world`.

import os
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Workaround for Gradio/pydantic bool-schema bug
# (https://github.com/gradio-app/gradio/issues/11722)
# Must run BEFORE importing gradio.
# ---------------------------------------------------------------------------
try:
    import gradio_client.utils as _gcu
    if hasattr(_gcu, "get_type"):
        _orig_get_type = _gcu.get_type
        def _patched_get_type(schema):
            if isinstance(schema, bool):
                return "bool"
            return _orig_get_type(schema)
        _gcu.get_type = _patched_get_type
    elif hasattr(_gcu, "_json_schema_to_python_type"):
        _orig_json_schema = _gcu._json_schema_to_python_type
        def _patched_json_schema(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            return _orig_json_schema(schema, defs) if defs is not None else _orig_json_schema(schema)
        _gcu._json_schema_to_python_type = _patched_json_schema
except Exception:
    pass
# ---------------------------------------------------------------------------

import cv2
import gradio as gr
import matplotlib
import numpy as np
import spaces
import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

print("[VGGT4D] Importing modules...", flush=True)
try:
    from vggt4d.masks.dynamic_mask import (
        adaptive_multiotsu_variance,
        cluster_attention_maps,
        extract_dyn_map,
    )
    from vggt4d.masks.refine_dyn_mask import RefineDynMask
    from vggt4d.models.vggt4d import VGGTFor4D
    from vggt4d.utils.model_utils import inference, organize_qk_dict
    from vggt.utils.load_fn import load_and_preprocess_images
    print("[VGGT4D] All imports OK", flush=True)
except Exception as e:
    print(f"[VGGT4D] IMPORT ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model loading — keep on CPU, ZeroGPU provides GPU only inside @spaces.GPU.
# ---------------------------------------------------------------------------
LOCAL_CKPT = "./ckpts/model_tracker_fixed_e20.pt"
CKPT_REPO = "facebook/VGGT_tracker_fixed"
CKPT_FILENAME = "model_tracker_fixed_e20.pt"

print("[VGGT4D] Loading checkpoint...", flush=True)
try:
    if os.path.exists(LOCAL_CKPT):
        ckpt_path = LOCAL_CKPT
    else:
        ckpt_path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILENAME)
    print(f"[VGGT4D] Checkpoint: {ckpt_path}", flush=True)

    model = VGGTFor4D()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    model.eval()
    print("[VGGT4D] Model loaded on CPU", flush=True)
except Exception as e:
    print(f"[VGGT4D] MODEL LOAD ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

MAX_FRAMES = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list[str]:
    """Extract frames from a video file, return list of temp image paths."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise gr.Error("Could not read video.")

    step = max(1, total // max_frames)
    tmpdir = tempfile.mkdtemp()
    paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and len(paths) < max_frames:
            p = os.path.join(tmpdir, f"frame_{len(paths):04d}.png")
            cv2.imwrite(p, frame)
            paths.append(p)
        idx += 1
    cap.release()

    if len(paths) < 2:
        raise gr.Error("Need at least 2 frames. Upload a longer video.")
    return paths


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert a single depth map to a colormapped uint8 RGB image."""
    d = depth.copy()
    lo, hi = np.percentile(d, [2, 98])
    d = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)
    cmap = matplotlib.colormaps["turbo"]
    colored = (cmap(d)[:, :, :3] * 255).astype(np.uint8)
    return colored


def mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary mask on an image in red."""
    vis = image.copy()
    vis[mask > 0] = (
        (1 - alpha) * vis[mask > 0] + alpha * np.array([255, 50, 50])
    ).astype(np.uint8)
    return vis


def images_to_video(frames: list[np.ndarray], fps: int = 8) -> str:
    """Write a list of RGB numpy frames to a temporary mp4 video."""
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(
        tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp.name


def pick_prediction_keys(predictions: dict, keys: tuple[str, ...]) -> dict:
    """Select only required prediction fields to minimize ZeroGPU payload size."""
    missing = [k for k in keys if k not in predictions]
    if missing:
        raise gr.Error(f"Model output missing required keys: {', '.join(missing)}")
    return {k: predictions[k] for k in keys}


# ---------------------------------------------------------------------------
# Pipeline stages — split into separate @spaces.GPU calls so each fits
# within ZeroGPU's time budget.
# ---------------------------------------------------------------------------

@spaces.GPU(duration=120)
def run_stage1(frame_paths):
    """Stage 1: predict depth and extract coarse dynamic map."""
    t0 = time.time()
    print(f"[VGGT4D] Stage 1 start - frames={len(frame_paths)}", flush=True)
    device = torch.device("cuda")
    model.to(device)

    images = load_and_preprocess_images(frame_paths).to(device)
    n_img, _, h_img, w_img = images.shape

    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(model, images)
    del agg_tokens_list
    qk_dict = organize_qk_dict(qk_dict, n_img)

    dyn_maps = extract_dyn_map(qk_dict, images)
    del qk_dict

    h_tok, w_tok = h_img // 14, w_img // 14
    feat_map = rearrange(
        enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok
    )
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
    del feat_map, enc_feat, dyn_maps

    upsampled_map = F.interpolate(
        rearrange(norm_dyn_map, "n h w -> n 1 h w"),
        size=(h_img, w_img),
        mode="bilinear",
        align_corners=False,
    )
    upsampled_map = rearrange(upsampled_map, "n 1 h w -> n h w")
    del norm_dyn_map

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = (upsampled_map > thres).cpu()
    del upsampled_map

    stage1_predictions = pick_prediction_keys(
        predictions1, ("depth", "intrinsic")
    )

    images_cpu = images.cpu()
    del images
    torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"[VGGT4D] Stage 1 done in {dt:.2f}s", flush=True)

    return images_cpu, stage1_predictions, dyn_masks


@spaces.GPU(duration=120)
def run_stage2(images_cpu, dyn_masks):
    """Stage 2: re-run inference with dynamic masks to refine camera poses."""
    t0 = time.time()
    print("[VGGT4D] Stage 2 start", flush=True)
    device = torch.device("cuda")
    model.to(device)

    images = images_cpu.to(device)
    predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))
    stage2_predictions = pick_prediction_keys(
        predictions2, ("extrinsic", "cam2world")
    )

    del images
    torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"[VGGT4D] Stage 2 done in {dt:.2f}s", flush=True)

    return stage2_predictions


@spaces.GPU(duration=120)
def run_stage3(images_cpu, dyn_masks, pred_depths, pred_cam2world, pred_intrinsic):
    """Stage 3: refine dynamic masks via geometric reprojection."""
    t0 = time.time()
    print("[VGGT4D] Stage 3 start", flush=True)
    device = torch.device("cuda")

    refiner = RefineDynMask(
        images_cpu.to(device),
        torch.tensor(pred_depths).to(device),
        dyn_masks.to(device),
        torch.tensor(pred_cam2world).float().to(device),
        torch.tensor(pred_intrinsic).to(device),
        device,
    )
    refined_mask = refiner.refine_masks().cpu()
    del refiner
    torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"[VGGT4D] Stage 3 done in {dt:.2f}s", flush=True)

    return refined_mask


# ---------------------------------------------------------------------------
# Orchestrator (runs on CPU, dispatches GPU stages)
# ---------------------------------------------------------------------------

def run_pipeline(video_path, enable_refinement):
    pipeline_t0 = time.time()
    if video_path is None:
        raise gr.Error("Please upload a video.")

    # --- Extract frames (CPU) ---
    frame_paths = extract_frames(video_path)
    print(f"[VGGT4D] Extracted {len(frame_paths)} frames", flush=True)

    # --- Stage 1 (GPU) ---
    try:
        images_cpu, stage1_predictions, dyn_masks = run_stage1(frame_paths)
    except Exception as e:
        raise gr.Error(
            f"Stage 1 failed ({type(e).__name__}). "
            "Possible cause: ZeroGPU timeout/OOM/payload issue. "
            "Try a shorter clip."
        ) from e

    n_img = images_cpu.shape[0]
    h_img, w_img = images_cpu.shape[2], images_cpu.shape[3]
    print(f"[VGGT4D] Resolution after preprocessing: {h_img}x{w_img}", flush=True)

    # --- Stage 2 (GPU) ---
    try:
        stage2_predictions = run_stage2(images_cpu, dyn_masks)
    except Exception as e:
        raise gr.Error(
            f"Stage 2 failed ({type(e).__name__}). "
            "Possible cause: ZeroGPU timeout/OOM/payload issue. "
            "Try a shorter clip or disable refinement."
        ) from e

    pred_depths = stage1_predictions["depth"]
    pred_cam2world = stage2_predictions["cam2world"]
    pred_intrinsic = stage1_predictions["intrinsic"]

    # --- Stage 3 (GPU, optional) ---
    if enable_refinement:
        try:
            refined_mask = run_stage3(
                images_cpu, dyn_masks, pred_depths, pred_cam2world, pred_intrinsic
            )
            masks_np = refined_mask.numpy().astype(bool)
        except Exception as e:
            print(
                f"[VGGT4D] Stage 3 failed ({type(e).__name__}): {e}. "
                "Falling back to coarse dynamic masks.",
                flush=True,
            )
            masks_np = dyn_masks.numpy().astype(bool)
    else:
        masks_np = dyn_masks.numpy().astype(bool)

    # --- Build visualisations (CPU) ---
    imgs_np = (
        images_cpu.permute(0, 2, 3, 1).numpy() * 255
    ).astype(np.uint8)
    depths_np = pred_depths

    depth_frames = [depth_to_colormap(depths_np[i]) for i in range(n_img)]
    overlay_frames = [mask_overlay(imgs_np[i], masks_np[i]) for i in range(n_img)]

    depth_video = images_to_video(depth_frames)
    mask_video = images_to_video(overlay_frames)

    gallery_items = []
    step = max(1, n_img // 8)
    for i in range(0, n_img, step):
        gallery_items.append((imgs_np[i], f"Frame {i}"))
        gallery_items.append((depth_frames[i], f"Depth {i}"))
        gallery_items.append((overlay_frames[i], f"Dynamic {i}"))

    total_dt = time.time() - pipeline_t0
    print(f"[VGGT4D] Pipeline done in {total_dt:.2f}s", flush=True)
    return depth_video, mask_video, gallery_items


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

css = """
.contain { max-width: 1200px; margin: auto; }
"""

with gr.Blocks(css=css, title="VGGT4D") as demo:
    gr.Markdown(
        """
        # VGGT4D — 4D Scene Understanding from Monocular Video
        Upload a short video to predict **depth maps**, **camera poses**, and **dynamic object masks**.

        The pipeline runs three stages:
        1. Predict depth and initial dynamic map from attention patterns
        2. Re-estimate camera poses masking dynamic regions
        3. *(Optional)* Refine dynamic masks via geometric consistency
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video")
            enable_stage3 = gr.Checkbox(
                label="Enable mask refinement (Stage 3)",
                value=False,
                info="Slower but more accurate dynamic masks. Disable to stay within GPU time limits.",
            )
            run_btn = gr.Button("Run VGGT4D", variant="primary")
            gr.Markdown(
                f"*Frames are subsampled to at most **{MAX_FRAMES}**. "
                "Short clips (2-5 s) work best to avoid ZeroGPU timeout.*"
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Depth Video"):
                    depth_out = gr.Video(label="Predicted Depth")
                with gr.TabItem("Dynamic Mask Video"):
                    mask_out = gr.Video(label="Dynamic Mask Overlay")
                with gr.TabItem("Gallery"):
                    gallery_out = gr.Gallery(
                        label="Sampled Frames",
                        columns=3,
                        height="auto",
                    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input, enable_stage3],
        outputs=[depth_out, mask_out, gallery_out],
    )

if __name__ == "__main__":
    demo.launch()

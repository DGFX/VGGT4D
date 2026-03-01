import os
import tempfile

import cv2
import gradio as gr
import matplotlib
import numpy as np
import spaces
import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

from vggt4d.masks.dynamic_mask import (
    adaptive_multiotsu_variance,
    cluster_attention_maps,
    extract_dyn_map,
)
from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt.utils.load_fn import load_and_preprocess_images

# ---------------------------------------------------------------------------
# Model loading — ZeroGPU intercepts .cuda()/.to('cuda') at module level
# and keeps weights on CPU until a @spaces.GPU function runs.
# ---------------------------------------------------------------------------
LOCAL_CKPT = "./ckpts/model_tracker_fixed_e20.pt"
CKPT_REPO = "facebook/VGGT_tracker_fixed"
CKPT_FILENAME = "model_tracker_fixed_e20.pt"

if os.path.exists(LOCAL_CKPT):
    ckpt_path = LOCAL_CKPT
else:
    ckpt_path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILENAME)

model = VGGTFor4D()
model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
model.eval()
model.cuda()

MAX_FRAMES = 50

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

    # Subsample if too many frames
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@spaces.GPU(duration=180)
def run_pipeline(video_path: str, progress=gr.Progress(track_tqdm=True)):
    if video_path is None:
        raise gr.Error("Please upload a video.")

    device = torch.device("cuda")

    # --- Extract frames ---
    progress(0, desc="Extracting frames...")
    frame_paths = extract_frames(video_path)
    n_frames = len(frame_paths)

    # --- Preprocess ---
    progress(0.05, desc=f"Preprocessing {n_frames} frames...")
    images = load_and_preprocess_images(frame_paths).to(device)
    n_img, _, h_img, w_img = images.shape

    # --- Stage 1: depth + dynamic map ---
    progress(0.10, desc="Stage 1: Predicting depth and dynamic map...")
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
    dyn_masks = upsampled_map > thres
    del upsampled_map

    # --- Stage 2: refine extrinsics ---
    progress(0.45, desc="Stage 2: Refining camera poses...")
    torch.cuda.empty_cache()
    predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))

    final_prediction = {**predictions1}
    final_prediction["extrinsic"] = predictions2["extrinsic"]
    final_prediction["cam2world"] = predictions2["cam2world"]
    del predictions1, predictions2

    # --- Stage 3: refine dynamic masks ---
    progress(0.70, desc="Stage 3: Refining dynamic masks...")
    torch.cuda.empty_cache()

    pred_depths = final_prediction["depth"]
    pred_cam2world = final_prediction["cam2world"]
    pred_intrinsic = final_prediction["intrinsic"]

    refiner = RefineDynMask(
        images,
        torch.tensor(pred_depths).to(device),
        dyn_masks.to(device),
        torch.tensor(pred_cam2world).float().to(device),
        torch.tensor(pred_intrinsic).to(device),
        device,
    )
    refined_mask = refiner.refine_masks()
    del refiner
    torch.cuda.empty_cache()

    # --- Build visualisations ---
    progress(0.90, desc="Building visualisations...")
    imgs_np = (
        images.cpu().permute(0, 2, 3, 1).numpy() * 255
    ).astype(np.uint8)  # (N, H, W, 3)
    masks_np = refined_mask.cpu().numpy().astype(bool)  # (N, H, W)
    depths_np = pred_depths  # (N, H, W)

    depth_frames = [depth_to_colormap(depths_np[i]) for i in range(n_img)]
    overlay_frames = [mask_overlay(imgs_np[i], masks_np[i]) for i in range(n_img)]

    depth_video = images_to_video(depth_frames)
    mask_video = images_to_video(overlay_frames)

    # Gallery images: interleaved input / depth / mask for first 8 frames
    gallery_items = []
    step = max(1, n_img // 8)
    for i in range(0, n_img, step):
        gallery_items.append((imgs_np[i], f"Frame {i}"))
        gallery_items.append((depth_frames[i], f"Depth {i}"))
        gallery_items.append((overlay_frames[i], f"Dynamic {i}"))

    progress(1.0, desc="Done!")
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
        3. Refine dynamic masks via geometric consistency
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video")
            run_btn = gr.Button("Run VGGT4D", variant="primary")
            gr.Markdown(
                f"*Frames are subsampled to at most **{MAX_FRAMES}** for memory. "
                "Short clips (2-10 s) work best.*"
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
        inputs=[video_input],
        outputs=[depth_out, mask_out, gallery_out],
    )

if __name__ == "__main__":
    demo.launch()

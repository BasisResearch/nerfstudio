"""
Code that uses VGGT (Visual Geometry Grounded deep Transformer)
to estimate camera poses and depth maps for structure from motion.

This module provides the following main functions:

1. run_vggt: Direct VGGT inference to COLMAP (depth-based reconstruction)
   - Runs VGGT model on images
   - Converts depth maps directly to 3D points
   - Outputs COLMAP format without bundle adjustment

2. run_vggt_ba: VGGT with feature-based tracking and bundle adjustment
   - Runs VGGT model on images for initial poses and depth
   - Uses VGGSfM feature tracking via predict_tracks()
   - Builds reconstruction with batch_np_matrix_to_pycolmap()
   - Refines with bundle adjustment
   - Outputs refined COLMAP format

3. refine_vggt_with_ba: (Legacy) Refines existing COLMAP reconstruction
   - Loads existing reconstruction and refines in-place
   - Use run_vggt_ba for the official VGGT approach

Requires:
- VGGT module from: https://github.com/facebookresearch/vggt
- PyTorch, PIL, pycolmap
"""

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from PIL import Image
import pycolmap

# Try to import vggt - it's an optional dependency
try:
    from vggt.dependency.track_predict import predict_tracks
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
    _HAS_VGGT = True
except ImportError:
    _HAS_VGGT = False
    predict_tracks = None  # type: ignore
    create_pixel_coordinate_grid = None  # type: ignore
    randomly_limit_trues = None  # type: ignore
    batch_np_matrix_to_pycolmap_wo_track = None  # type: ignore

from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import CONSOLE


def is_vggt_available() -> bool:
    """Check if VGGT is installed and available.

    Returns:
        True if VGGT is available, False otherwise.
    """
    return _HAS_VGGT


def run_vggt(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    conf_threshold: float = 50.0,
    mask_sky: bool = False,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    stride: int = 1,
    model_name: str = "facebook/VGGT-1B",
    shared_camera: bool = True,
) -> None:
    """Runs VGGT on images to estimate camera poses and depth (Facebook's feedforward mode).

    This follows Facebook's demo_colmap.py without bundle adjustment:
    VGGT inference → depth unprojection → filter points → batch_np_matrix_to_pycolmap_wo_track()

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use (not used by VGGT, kept for compatibility).
        verbose: If True, logs the output.
        conf_threshold: Confidence threshold (0-100%) for including points.
        mask_sky: If True, filter out points likely to be sky.
        mask_black_bg: If True, filter out points with very dark/black color.
        mask_white_bg: If True, filter out points with very bright/white color.
        stride: Stride for point sampling (higher = fewer points).
        model_name: HuggingFace model name for VGGT.
        shared_camera: If True, use single camera model for all frames.
    """
    if not _HAS_VGGT:
        CONSOLE.print(
            "[bold red]Error: To use vggt sfm_tool, you must install VGGT!\n"
            "Visit https://github.com/facebookresearch/vggt for installation instructions."
        )
        sys.exit(1)

    # Create output directory
    output_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run VGGT inference
    vggt_data = _run_vggt_inference(
        image_dir=image_dir,
        model_name=model_name,
        verbose=verbose,
    )

    # Extract data from inference results
    extrinsic = vggt_data["extrinsic"]
    intrinsic_downsampled = vggt_data["intrinsic_downsampled"]
    world_points = vggt_data["world_points"]
    colors_rgb = vggt_data["colors_rgb"]
    depth_conf = vggt_data["depth_conf"]
    image_paths = vggt_data["image_paths"]
    original_width = vggt_data["original_width"]
    original_height = vggt_data["original_height"]
    model_resolution = vggt_data["model_resolution"]

    # Filter and prepare 3D points in Facebook's format (points3d, points_xyf, points_rgb)
    # Note: Facebook uses conf_threshold as a value (e.g., 5.0), not percentile
    # For compatibility, we convert percentile to value if > 1
    if conf_threshold > 1.0:
        # Interpret as percentile and convert to value
        conf_threshold_value = np.percentile(depth_conf, conf_threshold)
    else:
        # Interpret as value directly
        conf_threshold_value = conf_threshold

    if verbose:
        CONSOLE.print(f"  - Using confidence threshold: {conf_threshold_value:.4f}")

    points3d, points_xyf, points_rgb = _filter_and_prepare_points_for_pycolmap(
        world_points=world_points,
        world_points_conf=depth_conf,
        colors_rgb=colors_rgb,
        conf_threshold=conf_threshold_value,
        stride=stride,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        verbose=verbose,
    )

    if verbose:
        CONSOLE.print(f"[bold yellow]Building pycolmap reconstruction at {model_resolution}x{model_resolution}...")

    # Step 1: Build reconstruction at model resolution (518x518) using intrinsic_downsampled
    reconstruction = _build_pycolmap_reconstruction_without_tracks(
        points3d=points3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsic=extrinsic,
        intrinsic=intrinsic_downsampled,
        image_paths=image_paths,
        image_size=(model_resolution, model_resolution),
        shared_camera=shared_camera,
        camera_type="SIMPLE_PINHOLE",
        verbose=verbose,
    )

    if reconstruction is None:
        CONSOLE.print("[bold red]Error: Failed to build pycolmap reconstruction!")
        sys.exit(1)

    # Step 2: Rescale reconstruction to original dimensions
    original_image_sizes = [(original_width, original_height)] * len(image_paths)
    reconstruction = _rescale_reconstruction_to_original_dimensions(
        reconstruction=reconstruction,
        image_paths=image_paths,
        original_image_sizes=original_image_sizes,
        model_resolution=model_resolution,
        shared_camera=shared_camera,
        verbose=verbose,
    )

    # Write reconstruction to binary format
    if verbose:
        CONSOLE.print(f"[bold yellow]Writing COLMAP files to {output_dir}")

    reconstruction.write_binary(str(output_dir))

    if verbose:
        CONSOLE.print(f"[bold green]✓ COLMAP reconstruction complete!")
        CONSOLE.print(f"  - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"  - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"  - 3D points: {len(reconstruction.points3D)}")


def run_vggt_ba(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    conf_threshold: float = 50.0,
    mask_sky: bool = False,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    stride: int = 1,
    model_name: str = "facebook/VGGT-1B",
    ba_refine_focal_length: bool = True,
    ba_refine_principal_point: bool = False,
    ba_refine_extra_params: bool = False,
    shared_camera: bool = True,
    max_query_pts: int = 2048,
    query_frame_num: int = 5,
    keypoint_extractor: str = "aliked+sp",
    max_points_num: int = 2048,
    fine_tracking: bool = True,
    vis_thresh: float = 0.5,
    max_reproj_error: Optional[float] = None,
    track_resolution: int = 518,
) -> None:
    """Runs VGGT with bundle adjustment refinement (Facebook's approach).

    This approach follows Facebook's demo_colmap.py:
    1. VGGT inference to get initial poses and depth
    2. Use feature-based tracking (VGGSfM) via predict_tracks() to get robust tracks
    3. Create pycolmap Reconstruction using batch_np_matrix_to_pycolmap()
    4. Run bundle adjustment to refine poses and intrinsics
    5. Write refined results to COLMAP format

    Note: Compatible with pycolmap>=0.4.0. For pycolmap>=0.6.0, all Path objects
    are explicitly converted to strings due to stricter type checking.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use (not used by VGGT, kept for compatibility).
        verbose: If True, logs the output.
        conf_threshold: Not used with track-based approach (kept for compatibility).
        mask_sky: Not used with track-based approach (kept for compatibility).
        mask_black_bg: Not used with track-based approach (kept for compatibility).
        mask_white_bg: Not used with track-based approach (kept for compatibility).
        stride: Not used with track-based approach (kept for compatibility).
        model_name: HuggingFace model name for VGGT.
        ba_refine_focal_length: If True, refine focal length during BA.
        ba_refine_principal_point: If True, refine principal point during BA.
        ba_refine_extra_params: If True, refine distortion params during BA.
        shared_camera: If True, use single camera model for all frames (like Facebook demo).
        max_query_pts: Maximum query points for track prediction. Lower values use less memory.
            Default: 2048. For memory-constrained GPUs, try 1024 or 512.
        query_frame_num: Number of query frames for track prediction. Lower values use less memory.
            Default: 5. For memory-constrained GPUs, try 3.
        keypoint_extractor: Keypoint extraction method ("aliked+sp", etc.).
        fine_tracking: Enable fine tracking in track prediction. Disabling significantly reduces memory.
            Default: True. For memory-constrained GPUs (e.g., RTX 4090 24GB), set to False.
            See: https://github.com/facebookresearch/vggt/issues/238
        vis_thresh: Visibility threshold for filtering tracks (0-1).
        max_reproj_error: Maximum reprojection error for track filtering (None = no limit).
        track_resolution: Resolution for loading images for track prediction.
    """
    if not _HAS_VGGT:
        CONSOLE.print(
            "[bold red]Error: To use vggt sfm_tool, you must install VGGT!\n"
            "Visit https://github.com/facebookresearch/vggt for installation instructions."
        )
        sys.exit(1)

    # Create output directory
    output_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run VGGT inference
    vggt_data = _run_vggt_inference(
        image_dir=image_dir,
        model_name=model_name,
        verbose=verbose,
    )

    # Extract data from inference results
    extrinsic = vggt_data["extrinsic"]
    intrinsic = vggt_data["intrinsic"]
    world_points = vggt_data["world_points"]
    colors_rgb = vggt_data["colors_rgb"]
    depth_conf = vggt_data["depth_conf"]
    image_paths = vggt_data["image_paths"]
    original_width = vggt_data["original_width"]
    original_height = vggt_data["original_height"]

    # Aggressively free memory after VGGT inference before track prediction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if verbose:
            CONSOLE.print(f"[bold yellow]Cleared GPU cache after VGGT inference")
            # Show current memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            CONSOLE.print(f"  - GPU memory allocated: {allocated:.2f} GiB")
            CONSOLE.print(f"  - GPU memory reserved: {reserved:.2f} GiB")

            # Check for RTX 4090 and warn about float32 auto-conversion issue
            gpu_name = torch.cuda.get_device_name(0)
            if "4090" in gpu_name or "RTX 4090" in gpu_name:
                CONSOLE.print(f"[bold yellow]Note: Detected RTX 4090 GPU")
                CONSOLE.print(f"  - RTX 4090 may auto-convert models to float32, using 2x memory")
                CONSOLE.print(f"  - See: https://github.com/facebookresearch/vggt/pull/253")

    # Use VGGT's native 518x518 resolution for tracking (much more memory efficient)
    # We'll scale the tracks to original resolution afterwards
    vggt_resolution = 518
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        CONSOLE.print(f"[bold yellow]Using resolution {vggt_resolution}x{vggt_resolution} for track prediction")
        CONSOLE.print(f"  - This uses ~4x less memory than 1024x1024")
        CONSOLE.print(f"  - Tracks will be scaled to original resolution after prediction")

    images = _load_images_for_tracking(image_paths, vggt_resolution, device, verbose)

    # Prepare depth and 3D points for track prediction at VGGT resolution
    depth_resized, points_3d, intrinsic_scaled = _prepare_depth_and_points_for_tracking(
        depth_conf=depth_conf,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        original_width=original_width,
        original_height=original_height,
        track_resolution=vggt_resolution,
        verbose=verbose,
    )

    if verbose:
        CONSOLE.print(f"[bold yellow]Running feature-based track prediction...")
        CONSOLE.print(f"  - max_query_pts: {max_query_pts}")
        CONSOLE.print(f"  - query_frame_num: {query_frame_num}")
        CONSOLE.print(f"  - fine_tracking: {fine_tracking}")
        CONSOLE.print(f"  - max_points_num: {max_points_num}")

        # Warn about memory usage if using high settings
        if fine_tracking or max_query_pts > 1024 or query_frame_num > 3:
            CONSOLE.print(f"[bold yellow]Memory usage tips:")
            if fine_tracking:
                CONSOLE.print(f"  - fine_tracking=True uses significant memory. Set to False if OOM occurs.")
            if max_query_pts > 1024:
                CONSOLE.print(f"  - max_query_pts={max_query_pts} is high. Try 1024 or 512 if OOM occurs.")
            if query_frame_num > 3:
                CONSOLE.print(f"  - query_frame_num={query_frame_num} is high. Try 3 if OOM occurs.")
            CONSOLE.print(f"  - See: https://github.com/facebookresearch/vggt/issues/238")

    # Predict tracks using VGGSfM
    # Note: predict_tracks expects conf and points_3d as NUMPY ARRAYS, not torch tensors
    # Only images should be a torch tensor (following Facebook demo pattern)
    if verbose:
        CONSOLE.print(f"  - images shape for predict_tracks: {images.shape}")
        CONSOLE.print(f"  - depth_resized shape for predict_tracks: {depth_resized.shape}")
        CONSOLE.print(f"  - points_3d shape for predict_tracks: {points_3d.shape}")

    pred_tracks, pred_vis_scores, pred_confs, refined_points_3d, points_rgb = predict_tracks(
        images=images,
        conf=depth_resized,
        points_3d=points_3d,
        masks=None,
        max_query_pts=max_query_pts,
        query_frame_num=query_frame_num,
        keypoint_extractor=keypoint_extractor,
        max_points_num=max_points_num,
        fine_tracking=fine_tracking,
    )

    # Free GPU memory from input tensors immediately
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert to numpy and free GPU memory
    if isinstance(pred_tracks, torch.Tensor):
        pred_tracks = pred_tracks.cpu().numpy()
    if isinstance(pred_vis_scores, torch.Tensor):
        pred_vis_scores = pred_vis_scores.cpu().numpy()
    if isinstance(refined_points_3d, torch.Tensor):
        refined_points_3d = refined_points_3d.cpu().numpy()
    if isinstance(points_rgb, torch.Tensor):
        points_rgb = points_rgb.cpu().numpy()

    # Clear GPU cache after converting all tensors to CPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Filter tracks by visibility threshold
    track_mask = pred_vis_scores > vis_thresh

    if verbose:
        CONSOLE.print(f"  - Predicted tracks shape: {pred_tracks.shape}")
        CONSOLE.print(f"  - Visibility scores shape: {pred_vis_scores.shape}")
        CONSOLE.print(f"  - Valid tracks: {np.sum(track_mask)}/{track_mask.size}")
        CONSOLE.print(f"[bold yellow]Building pycolmap reconstruction from tracks...")

    # Build pycolmap reconstruction using batch_np_matrix_to_pycolmap
    reconstruction = _build_pycolmap_reconstruction_from_tracks(
        points3D=refined_points_3d,
        extrinsic=extrinsic,
        intrinsic=intrinsic_scaled,
        tracks=pred_tracks,
        image_paths=image_paths,
        image_size=(track_resolution, track_resolution),
        masks=track_mask,
        shared_camera=shared_camera,
        max_reproj_error=max_reproj_error,
        points_rgb=points_rgb,
        verbose=verbose,
    )

    if reconstruction is None:
        CONSOLE.print("[bold red]Error: Failed to build pycolmap reconstruction!")
        sys.exit(1)

    if verbose:
        CONSOLE.print(f"  - Reconstruction built with {len(reconstruction.points3D)} points")
        CONSOLE.print(f"[bold yellow]Running bundle adjustment...")

    # Run bundle adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = ba_refine_focal_length
    ba_options.refine_principal_point = ba_refine_principal_point
    ba_options.refine_extra_params = ba_refine_extra_params

    pycolmap.bundle_adjustment(reconstruction, ba_options)

    if verbose:
        CONSOLE.print(f"[bold green]✓ Bundle adjustment complete")

    # Rescale reconstruction from track_resolution to original dimensions
    original_image_sizes = [(original_width, original_height)] * len(image_paths)
    reconstruction = _rescale_reconstruction_to_original_dimensions(
        reconstruction=reconstruction,
        image_paths=image_paths,
        original_image_sizes=original_image_sizes,
        model_resolution=track_resolution,
        shared_camera=shared_camera,
        verbose=verbose,
    )

    # Write refined COLMAP files
    if verbose:
        CONSOLE.print(f"[bold yellow]Writing refined COLMAP files to {output_dir}")

    reconstruction.write_binary(str(output_dir))

    if verbose:
        CONSOLE.print(f"[bold green]✓ COLMAP reconstruction with BA complete!")
        CONSOLE.print(f"  - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"  - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"  - 3D points: {len(reconstruction.points3D)}")


# ============================================================================
# SHARED HELPER FUNCTIONS
# ============================================================================

def _run_vggt_inference(
    image_dir: Path,
    model_name: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Run VGGT inference and return all necessary data.

    This is the shared inference logic used by both run_vggt and run_vggt_ba.

    Returns:
        Dictionary containing:
        - extrinsic: Camera extrinsic matrices
        - intrinsic: Camera intrinsic matrices
        - world_points: 3D points from depth
        - colors_rgb: RGB colors for points
        - depth_conf: Depth confidence values
        - image_paths: Paths to input images
        - original_width/height: Original image dimensions
    """
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except ImportError:
        CONSOLE.print(
            "[bold red]Error: To use VGGT, you must install it!\n"
            "Install from: https://github.com/facebookresearch/vggt\n"
            "Example: pip install git+https://github.com/facebookresearch/vggt.git"
        )
        sys.exit(1)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        CONSOLE.print(f"[bold green]Using device: {device}")

    # Load VGGT model
    if verbose:
        CONSOLE.print(f"[bold yellow]Loading VGGT model: {model_name}")

    model = VGGT.from_pretrained(model_name)
    model.eval()
    model = model.to(device)

    # Get image paths
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    if len(image_paths) == 0:
        CONSOLE.print(f"[bold red]Error: No images found in {image_dir}")
        sys.exit(1)

    if verbose:
        CONSOLE.print(f"[bold green]Found {len(image_paths)} images")

    # Load original images for RGB colors
    original_images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))

    original_height, original_width = original_images[0].shape[:2]

    # Load and preprocess images for VGGT
    image_names = [str(p) for p in image_paths]
    images = load_and_preprocess_images(image_names).to(device)

    if verbose:
        CONSOLE.print(f"[bold yellow]Running VGGT inference...")
        CONSOLE.print(f"  - Images shape after preprocessing: {images.shape}")

    # Run inference
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to camera parameters
    # Store image shape before freeing the tensor
    image_shape = images.shape[-2:]

    # Free images tensor and model from GPU immediately after inference
    del images
    # Move model to CPU to free GPU memory
    if torch.cuda.is_available():
        model = model.cpu()
        del model
        torch.cuda.empty_cache()

    extrinsic, intrinsic_downsampled = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_shape
    )
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], [original_height, original_width]
    )

    # Move predictions to CPU and convert to numpy
    # Do this in a single pass to minimize GPU operations
    if verbose:
        CONSOLE.print(f"[bold yellow]Converting tensors to numpy...")
        CONSOLE.print(f"  - predictions keys: {list(predictions.keys())}")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                CONSOLE.print(f"  - {key} shape before squeeze: {predictions[key].shape}")

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            pred_np = predictions[key].cpu().numpy()
            # Only squeeze if the first dimension is actually size 1 (batch dimension)
            if pred_np.shape[0] == 1:
                predictions[key] = np.squeeze(pred_np, axis=0)
            else:
                predictions[key] = pred_np
                if verbose:
                    CONSOLE.print(f"[bold yellow]Warning: {key} first dimension is {pred_np.shape[0]}, not squeezing")

    if verbose:
        CONSOLE.print(f"  - extrinsic shape before squeeze: {extrinsic.shape}")
        CONSOLE.print(f"  - intrinsic shape before squeeze: {intrinsic.shape}")
        CONSOLE.print(f"  - intrinsic_downsampled shape before squeeze: {intrinsic_downsampled.shape}")

    # Convert to numpy and handle batch dimension
    # If shape is (1, N, ...), squeeze removes the batch dimension to get (N, ...)
    # If shape is already (N, ...), squeezing axis 0 would fail if N > 1
    extrinsic_np = extrinsic.cpu().numpy()
    intrinsic_np = intrinsic.cpu().numpy()
    intrinsic_downsampled_np = intrinsic_downsampled.cpu().numpy()

    # Only squeeze if the first dimension is actually size 1 (batch dimension)
    if extrinsic_np.shape[0] == 1:
        extrinsic = np.squeeze(extrinsic_np, axis=0)
    else:
        extrinsic = extrinsic_np
        if verbose:
            CONSOLE.print(f"[bold yellow]Warning: extrinsic first dimension is {extrinsic_np.shape[0]}, not squeezing")

    if intrinsic_np.shape[0] == 1:
        intrinsic = np.squeeze(intrinsic_np, axis=0)
    else:
        intrinsic = intrinsic_np
        if verbose:
            CONSOLE.print(f"[bold yellow]Warning: intrinsic first dimension is {intrinsic_np.shape[0]}, not squeezing")

    if intrinsic_downsampled_np.shape[0] == 1:
        intrinsic_downsampled = np.squeeze(intrinsic_downsampled_np, axis=0)
    else:
        intrinsic_downsampled = intrinsic_downsampled_np
        if verbose:
            CONSOLE.print(f"[bold yellow]Warning: intrinsic_downsampled first dimension is {intrinsic_downsampled_np.shape[0]}, not squeezing")

    # Clear GPU cache after all conversions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute 3D points from depth maps
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic_downsampled)

    if verbose:
        CONSOLE.print(f"[bold green]✓ VGGT inference complete")
        CONSOLE.print(f"  - Camera poses: {extrinsic.shape}")
        CONSOLE.print(f"  - 3D points: {world_points.shape}")

    # Prepare RGB colors at the same resolution as depth
    S, H, W = world_points.shape[:3]
    normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)

    for i, img in enumerate(original_images):
        resized_img = cv2.resize(img, (W, H))
        normalized_images[i] = resized_img / 255.0

    colors_rgb = (normalized_images * 255).astype(np.uint8)

    # Get depth confidence
    depth_conf = predictions.get("depth_conf", np.ones_like(depth_map[..., 0]))

    # Determine model resolution from world_points shape
    model_resolution = world_points.shape[1]  # H dimension (should be 518)

    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "intrinsic_downsampled": intrinsic_downsampled,
        "world_points": world_points,
        "colors_rgb": colors_rgb,
        "depth_conf": depth_conf,
        "image_paths": image_paths,
        "original_width": original_width,
        "original_height": original_height,
        "model_resolution": model_resolution,
    }



def _build_pycolmap_reconstruction_from_tracks(
    points3D: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    tracks: np.ndarray,
    image_paths: List[Path],
    image_size: Tuple[int, int],
    masks: Optional[np.ndarray] = None,
    shared_camera: bool = True,
    max_reproj_error: Optional[float] = None,
    points_rgb: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    """Build a pycolmap Reconstruction from feature-based tracks using VGGT's utility.

    This is a wrapper around batch_np_matrix_to_pycolmap from VGGT, which uses
    feature-based tracking (VGGSfM) instead of depth-based tracks.

    Args:
        points3D: (P, 3) array of 3D point coordinates
        extrinsic: Camera extrinsic matrices (N, 4, 4)
        intrinsic: Camera intrinsic matrices (N, 3, 3)
        tracks: (N, P, 2) array of 2D point projections across frames
        image_paths: Paths to images
        image_size: (width, height) tuple
        masks: Optional (N, P) boolean array indicating valid tracks
        shared_camera: If True, use single camera model for all frames
        max_reproj_error: Optional threshold for reprojection error filtering
        points_rgb: Optional (P, 3) RGB color values for points
        verbose: If True, log progress

    Returns:
        pycolmap.Reconstruction object or None if failed
    """
    try:
        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
    except ImportError:
        CONSOLE.print(
            "[bold red]Error: VGGT dependency not found!\n"
            "batch_np_matrix_to_pycolmap requires: https://github.com/facebookresearch/vggt"
        )
        return None

    # Note: extrinsic from pose_encoding_to_extri_intri is already (N, 3, 4) format
    # batch_np_matrix_to_pycolmap expects (N, 3, 4) extrinsics
    if extrinsic.shape[1] == 4 and extrinsic.shape[2] == 4:
        # If somehow we have (N, 4, 4), convert to (N, 3, 4)
        extrinsic_3x4 = extrinsic[:, :3, :]
    elif extrinsic.shape[1] == 3 and extrinsic.shape[2] == 4:
        # Already in correct (N, 3, 4) format
        extrinsic_3x4 = extrinsic
    else:
        raise ValueError(f"Unexpected extrinsic shape: {extrinsic.shape}. Expected (N, 3, 4) or (N, 4, 4)")

    # Convert image_size from (width, height) to array
    image_size_array = np.array([image_size[0], image_size[1]])

    # Determine camera type based on shared_camera setting
    camera_type = "SIMPLE_PINHOLE" if shared_camera else "PINHOLE"

    if verbose:
        CONSOLE.print(f"  - Calling batch_np_matrix_to_pycolmap with {len(points3D)} 3D points")
        CONSOLE.print(f"  - Track shape: {tracks.shape}")
        CONSOLE.print(f"  - Camera type: {camera_type}")

    # Call VGGT's batch_np_matrix_to_pycolmap
    reconstruction, valid_mask = batch_np_matrix_to_pycolmap(
        points3d=points3D,
        extrinsics=extrinsic_3x4,
        intrinsics=intrinsic,
        tracks=tracks,
        image_size=image_size_array,
        masks=masks,
        max_reproj_error=max_reproj_error,
        shared_camera=shared_camera,
        camera_type=camera_type,
        points_rgb=points_rgb,
    )

    if reconstruction is None:
        if verbose:
            CONSOLE.print("[bold yellow]Warning: batch_np_matrix_to_pycolmap returned None")
        return None

    # Add image names to reconstruction
    for i, img_path in enumerate(image_paths):
        if (i + 1) in reconstruction.images:
            reconstruction.images[i + 1].name = img_path.name

    if verbose:
        CONSOLE.print(f"  - Created reconstruction:")
        CONSOLE.print(f"    - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"    - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"    - Points3D: {len(reconstruction.points3D)}")
        CONSOLE.print(f"    - Valid tracks: {np.sum(valid_mask)}/{len(valid_mask)}")

    return reconstruction


def _load_images_for_tracking(
    image_paths: List[Path],
    resolution: int,
    device: str,
    verbose: bool = False,
):
    """Load and preprocess images at specified resolution for track prediction.

    Args:
        image_paths: List of paths to images
        resolution: Target resolution (images will be resized to resolution x resolution)
        device: Device to load tensors to ("cuda" or "cpu")
        verbose: If True, log progress

    Returns:
        Tensor of shape (N, 3, H, W) with images normalized to [0, 1]
    """
    try:
        import torch
        from PIL import Image
    except ImportError:
        CONSOLE.print("[bold red]Error: torch and PIL are required!")
        sys.exit(1)

    if verbose:
        CONSOLE.print(f"[bold yellow]Loading {len(image_paths)} images at resolution {resolution}...")

    # Use pinned memory for faster CPU->GPU transfers if using CUDA
    use_pinned = device == "cuda"

    images_list = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((resolution, resolution), Image.BILINEAR)
        # Create contiguous tensor for better memory layout
        img_array = np.array(img, dtype=np.float32)  # Use float32 directly
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 255.0

        if use_pinned:
            img_tensor = img_tensor.pin_memory()

        images_list.append(img_tensor)

    # Stack and move to device efficiently
    images = torch.stack(images_list)
    if device == "cuda":
        images = images.to(device, non_blocking=True).contiguous()
    else:
        images = images.to(device).contiguous()

    if verbose:
        CONSOLE.print(f"  - Loaded images with shape: {images.shape}")

    return images


def _prepare_depth_and_points_for_tracking(
    depth_conf: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    original_width: int,
    original_height: int,
    track_resolution: int,
    verbose: bool = False,
):
    """Resize depth maps and compute 3D points for track prediction.

    Args:
        depth_conf: Depth confidence maps from VGGT (N, H, W)
        extrinsic: Camera extrinsic matrices (N, 4, 4)
        intrinsic: Camera intrinsic matrices (N, 3, 3) at original resolution
        original_width: Original image width
        original_height: Original image height
        track_resolution: Target resolution for tracking
        verbose: If True, log progress

    Returns:
        Tuple of:
        - depth_resized: Resized depth maps (N, track_resolution, track_resolution)
        - points_3d: 3D points from unprojection (N, track_resolution, track_resolution, 3)
        - intrinsic_scaled: Scaled intrinsic matrices (N, 3, 3)
    """
    try:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except ImportError:
        CONSOLE.print("[bold red]Error: VGGT geometry utils not found!")
        sys.exit(1)

    num_images = len(depth_conf)

    # Resize depth to track_resolution
    # Use C-contiguous array for better memory access patterns
    depth_resized = np.zeros((num_images, track_resolution, track_resolution), dtype=np.float32, order='C')
    for i in range(num_images):
        depth_resized[i] = cv2.resize(depth_conf[i], (track_resolution, track_resolution))

    # Scale intrinsics from original resolution to track_resolution
    # Use max dimension for scaling to preserve aspect ratio
    scale_factor = track_resolution / max(original_width, original_height)
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[:, :2, :] *= scale_factor

    # Ensure contiguous arrays for better memory access
    depth_resized = np.ascontiguousarray(depth_resized)

    # Add trailing dimension for unproject_depth_map_to_point_map
    # The function expects shape (N, H, W, 1) to handle the .squeeze(-1) operation
    depth_resized_with_channel = depth_resized[:, :, :, np.newaxis]

    # Unproject depth to get 3D points
    points_3d = unproject_depth_map_to_point_map(depth_resized_with_channel, extrinsic, intrinsic_scaled)

    if verbose:
        CONSOLE.print(f"  - Resized depth to: {depth_resized.shape}")
        CONSOLE.print(f"  - Computed 3D points: {points_3d.shape}")
        CONSOLE.print(f"  - Scaled intrinsics by factor: {scale_factor:.4f}")

    return depth_resized, points_3d, intrinsic_scaled


# ============================================================================
# LEGACY FUNCTION (for backward compatibility)
# ============================================================================

def refine_vggt_with_ba(
    colmap_dir: Path,
    verbose: bool = False,
) -> None:
    """Refine VGGT reconstruction with bundle adjustment using pycolmap.

    This function loads an existing VGGT COLMAP reconstruction and refines it
    using bundle adjustment.

    NOTE: This is a simplified approach that refines an existing reconstruction
    in-place. For the official VGGT approach with feature-based tracking (VGGSfM),
    use run_vggt_ba() instead, which calls predict_tracks() and
    batch_np_matrix_to_pycolmap() as in the official demo_colmap.py.

    Args:
        colmap_dir: Path to the COLMAP directory containing sparse/0
        verbose: If True, logs the output.
    """
    try:
        import pycolmap
    except ImportError:
        CONSOLE.print("[bold red]Error: pycolmap is required for bundle adjustment!")
        CONSOLE.print("Install with: pip install pycolmap")
        sys.exit(1)

    sparse_dir = colmap_dir / "sparse" / "0"

    if not sparse_dir.exists():
        CONSOLE.print(f"[bold red]Error: COLMAP reconstruction not found at {sparse_dir}")
        sys.exit(1)

    if verbose:
        CONSOLE.print(f"[bold yellow]Running bundle adjustment on VGGT reconstruction...")

    # Load existing reconstruction
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))

    if verbose:
        CONSOLE.print(f"  - Loaded: {len(reconstruction.cameras)} cameras, "
                     f"{len(reconstruction.images)} images, "
                     f"{len(reconstruction.points3D)} points")

    # Run bundle adjustment with default options (refine focal length only)
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = True
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False

    pycolmap.bundle_adjustment(reconstruction, ba_options)

    # Write refined reconstruction
    reconstruction.write(str(sparse_dir))

    if verbose:
        CONSOLE.print(f"[bold green]✓ Bundle adjustment complete")
        CONSOLE.print(f"  - Refined reconstruction written to {sparse_dir}")


# ============================================================================
# FACEBOOK VGGT HELPER FUNCTIONS
# These functions are sourced from Facebook's VGGT repository to match their
# exact post-model workflow for COLMAP reconstruction.
# ============================================================================

def _build_pycolmap_reconstruction_without_tracks(
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    points_rgb: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_paths: List[Path],
    image_size: Tuple[int, int],
    shared_camera: bool,
    camera_type: str,
    verbose: bool,
) -> Optional[Any]:
    """Build pycolmap reconstruction without tracks using Facebook's function.

    This is a thin wrapper around Facebook's batch_np_matrix_to_pycolmap_wo_track function.

    Args:
        points3d: 3D points (P, 3)
        points_xyf: 2D points with frame indices (P, 3) - [x, y, frame_idx]
        points_rgb: RGB colors (P, 3)
        extrinsic: Camera extrinsics (N, 4, 4)
        intrinsic: Camera intrinsics (N, 3, 3)
        image_paths: Paths to images
        image_size: (width, height) of images
        shared_camera: Whether to use a single shared camera
        camera_type: Camera model type (e.g., "SIMPLE_PINHOLE")
        verbose: Whether to print progress

    Returns:
        pycolmap Reconstruction object, or None if failed
    """
    # Convert extrinsic from (N, 4, 4) to (N, 3, 4) if needed
    if extrinsic.shape[1] == 4:
        extrinsic_3x4 = extrinsic[:, :3, :]
    else:
        extrinsic_3x4 = extrinsic

    # image_size should be [width, height] for Facebook's function
    image_size_array = np.array([image_size[0], image_size[1]])

    if verbose:
        CONSOLE.print(f"  - Calling batch_np_matrix_to_pycolmap_wo_track with {len(points3d)} 3D points")
        CONSOLE.print(f"  - points_xyf shape: {points_xyf.shape}")
        CONSOLE.print(f"  - Camera type: {camera_type}")

    # Call Facebook's function
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points3d=points3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsics=extrinsic_3x4,
        intrinsics=intrinsic,
        image_size=image_size_array,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    if reconstruction is None:
        if verbose:
            CONSOLE.print("[bold yellow]Warning: batch_np_matrix_to_pycolmap_wo_track returned None")
        return None

    # Update image names to actual filenames
    for i, img_path in enumerate(image_paths):
        if (i + 1) in reconstruction.images:
            reconstruction.images[i + 1].name = img_path.name

    if verbose:
        CONSOLE.print(f"  - Created reconstruction:")
        CONSOLE.print(f"    - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"    - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"    - Points3D: {len(reconstruction.points3D)}")

    return reconstruction


def _rescale_reconstruction_to_original_dimensions(
    reconstruction: Any,
    image_paths: List[Path],
    original_image_sizes: List[Tuple[int, int]],
    model_resolution: int,
    shared_camera: bool,
    verbose: bool,
) -> Any:
    """Rescale reconstruction from model resolution to original dimensions.

    This is based on Facebook's rename_colmap_recons_and_rescale_camera function.
    VGGT builds reconstructions at 518x518 resolution, so we need to rescale
    camera parameters and image dimensions to match the original images.

    Args:
        reconstruction: pycolmap Reconstruction object
        image_paths: Paths to images
        original_image_sizes: List of (width, height) tuples for each image
        model_resolution: Resolution used by VGGT model (518)
        shared_camera: Whether using a single shared camera
        verbose: Whether to print progress

    Returns:
        Updated pycolmap Reconstruction object
    """
    import copy

    if verbose:
        CONSOLE.print(f"[bold yellow]Rescaling reconstruction from {model_resolution}x{model_resolution} to original dimensions")

    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]

        # Rename image to original filename
        pyimage.name = image_paths[pyimageid - 1].name

        if rescale_camera:
            # Get original image dimensions
            orig_width, orig_height = original_image_sizes[pyimageid - 1]
            real_image_size = np.array([orig_width, orig_height])

            # Calculate resize ratio
            resize_ratio = max(real_image_size) / model_resolution

            # Rescale camera parameters
            pred_params = copy.deepcopy(pycamera.params)
            pred_params = pred_params * resize_ratio

            # Set principal point to center of original image
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            # Update camera
            pycamera.params = pred_params
            pycamera.width = int(orig_width)
            pycamera.height = int(orig_height)

            if shared_camera:
                # If shared camera, only need to rescale once
                rescale_camera = False

    if verbose:
        CONSOLE.print(f"[bold green]✓ Rescaled reconstruction to original dimensions")

    return reconstruction


def _filter_and_prepare_points_for_pycolmap(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    colors_rgb: np.ndarray,
    conf_threshold: float,
    stride: int,
    mask_black_bg: bool,
    mask_white_bg: bool,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter points using Facebook's exact approach for feedforward reconstruction.

    This function applies confidence filtering and random sampling to prepare points
    for batch_np_matrix_to_pycolmap_wo_track, following Facebook's demo_colmap.py.

    Args:
        world_points: 3D points from depth unprojection (S, H, W, 3)
        world_points_conf: Confidence values for each point (S, H, W)
        colors_rgb: RGB colors for each point (S, H, W, 3)
        conf_threshold: Confidence threshold value (not percentile) for filtering
        stride: Sampling stride for points (higher = fewer points)
        mask_black_bg: If True, filter out very dark points
        mask_white_bg: If True, filter out very bright points
        verbose: If True, log progress

    Returns:
        Tuple of:
        - points3d: Filtered 3D points (P, 3)
        - points_xyf: Pixel coordinates with frame indices (P, 3) - [x, y, frame_idx]
        - points_rgb: RGB colors (P, 3)
    """
    S, H, W = world_points.shape[:3]

    if verbose:
        CONSOLE.print(f"[bold yellow]Filtering points using Facebook's approach...")
        CONSOLE.print(f"  - Input shape: {world_points.shape}")

    # Create pixel coordinate grid (using Facebook's function directly)
    points_xyf = create_pixel_coordinate_grid(S, H, W)

    # Apply confidence threshold (Facebook uses value threshold, not percentile)
    conf_mask = world_points_conf >= conf_threshold

    # Apply color masks if requested
    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=-1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        white_bg_mask = ~(
            (colors_rgb[..., 0] > 240) &
            (colors_rgb[..., 1] > 240) &
            (colors_rgb[..., 2] > 240)
        )
        conf_mask = conf_mask & white_bg_mask

    # Apply stride (subsample points)
    if stride > 1:
        stride_mask = np.zeros((H, W), dtype=bool)
        stride_mask[::stride, ::stride] = True
        stride_mask = np.broadcast_to(stride_mask[np.newaxis, :, :], (S, H, W))
        conf_mask = conf_mask & stride_mask

    if verbose:
        CONSOLE.print(f"  - Points after confidence & mask filtering: {np.sum(conf_mask):,}")

    # Limit to max 100k points (using Facebook's function directly)
    conf_mask = randomly_limit_trues(conf_mask, 100000)

    if verbose:
        CONSOLE.print(f"  - Points after random limiting (max 100k): {np.sum(conf_mask):,}")

    # Filter points
    points3d = world_points[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = colors_rgb[conf_mask]

    if verbose:
        CONSOLE.print(f"[bold green]✓ Filtered to {len(points3d):,} points")

    return points3d, points_xyf, points_rgb



def _hash_point(point: np.ndarray, scale: float = 100) -> int:
    """Create a hash for a 3D point by quantizing coordinates.

    This is used to merge nearby 3D points that are effectively the same point
    observed from multiple views. The scale parameter controls the precision
    of the quantization (higher = finer precision).

    Args:
        point: 3D point coordinates (3,)
        scale: Quantization scale factor (default: 100)

    Returns:
        Hash value for the quantized point
    """
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)





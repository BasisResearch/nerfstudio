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
    _HAS_VGGT = True
except ImportError:
    _HAS_VGGT = False
    predict_tracks = None  # type: ignore

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
) -> None:
    """Runs VGGT on images to estimate camera poses and depth (jckhng's approach).

    This is the simple, direct approach: VGGT inference → depth unprojection → COLMAP.
    No bundle adjustment is performed.

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

    # Convert camera poses to COLMAP format
    quaternions, translations = _extrinsic_to_colmap_format(extrinsic)

    # Filter and prepare 3D points
    points3D, image_points2D = _filter_and_prepare_points(
        world_points=world_points,
        world_points_conf=depth_conf,
        colors_rgb=colors_rgb,
        conf_threshold=conf_threshold,
        stride=stride,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        verbose=verbose,
    )

    # Write COLMAP files (binary format)
    if verbose:
        CONSOLE.print(f"[bold yellow]Writing COLMAP files to {output_dir}")

    _write_colmap_cameras_bin(output_dir / "cameras.bin", intrinsic, original_width, original_height)
    _write_colmap_images_bin(output_dir / "images.bin", quaternions, translations, image_points2D, image_paths)
    _write_colmap_points3D_bin(output_dir / "points3D.bin", points3D)

    if verbose:
        CONSOLE.print(f"[bold green]✓ COLMAP reconstruction complete!")
        CONSOLE.print(f"  - Cameras: {len(intrinsic)}")
        CONSOLE.print(f"  - Images: {len(quaternions)}")
        CONSOLE.print(f"  - 3D points: {len(points3D)}")


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

    # Extract refined data from reconstruction
    refined_quaternions, refined_translations, refined_intrinsics, refined_points3D, refined_image_points2D = \
        _extract_from_pycolmap_reconstruction(reconstruction, image_paths)

    # Rescale intrinsics and 2D points from track_resolution to original resolution
    scale_x = original_width / track_resolution
    scale_y = original_height / track_resolution

    for i in range(len(refined_intrinsics)):
        refined_intrinsics[i][0, 0] *= scale_x  # fx
        refined_intrinsics[i][1, 1] *= scale_y  # fy
        refined_intrinsics[i][0, 2] *= scale_x  # cx
        refined_intrinsics[i][1, 2] *= scale_y  # cy

    # Rescale 2D points
    for i in range(len(refined_image_points2D)):
        for j in range(len(refined_image_points2D[i])):
            x, y, point3d_id = refined_image_points2D[i][j]
            refined_image_points2D[i][j] = (x * scale_x, y * scale_y, point3d_id)

    # Write COLMAP files (binary format) with refined results
    if verbose:
        CONSOLE.print(f"[bold yellow]Writing refined COLMAP files to {output_dir}")

    _write_colmap_cameras_bin(output_dir / "cameras.bin", refined_intrinsics, original_width, original_height)
    _write_colmap_images_bin(output_dir / "images.bin", refined_quaternions, refined_translations,
                             refined_image_points2D, image_paths)
    _write_colmap_points3D_bin(output_dir / "points3D.bin", refined_points3D)

    if verbose:
        CONSOLE.print(f"[bold green]✓ COLMAP reconstruction with BA complete!")
        CONSOLE.print(f"  - Cameras: {len(refined_intrinsics)}")
        CONSOLE.print(f"  - Images: {len(refined_quaternions)}")
        CONSOLE.print(f"  - 3D points: {len(refined_points3D)}")


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

    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "world_points": world_points,
        "colors_rgb": colors_rgb,
        "depth_conf": depth_conf,
        "image_paths": image_paths,
        "original_width": original_width,
        "original_height": original_height,
    }


def _build_pycolmap_reconstruction(
    points3D: List[Dict],
    image_points2D: List[List],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_paths: List[Path],
    image_size: Tuple[int, int],
    shared_camera: bool,
    verbose: bool,
):
    """Build a pycolmap Reconstruction from depth-based tracks.

    This follows the pattern from Facebook's batch_np_matrix_to_pycolmap function.

    Args:
        points3D: List of 3D point dictionaries with tracks
        image_points2D: List of 2D points per image
        extrinsic: Camera extrinsic matrices (N, 4, 4)
        intrinsic: Camera intrinsic matrices (N, 3, 3)
        image_paths: Paths to images
        image_size: (width, height) tuple
        shared_camera: If True, use single camera model for all frames
        verbose: If True, log progress

    Returns:
        pycolmap.Reconstruction object or None if failed
    """
    try:
        import pycolmap
    except ImportError:
        return None

    reconstruction = pycolmap.Reconstruction()
    width, height = image_size

    # Add cameras
    if shared_camera:
        # Use single camera for all images (like Facebook demo)
        # Average the intrinsics
        mean_intrinsic = np.mean(intrinsic, axis=0)
        fx = float(mean_intrinsic[0, 0])
        fy = float(mean_intrinsic[1, 1])
        cx = float(mean_intrinsic[0, 2])
        cy = float(mean_intrinsic[1, 2])

        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[fx, cx, cy],  # SIMPLE_PINHOLE uses average of fx/fy
        )
        camera.camera_id = 1
        reconstruction.add_camera(camera)
        camera_ids = [1] * len(image_paths)
    else:
        # Create separate camera for each image
        camera_ids = []
        for i, K in enumerate(intrinsic):
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])

            camera = pycolmap.Camera(
                model="PINHOLE",
                width=width,
                height=height,
                params=[fx, fy, cx, cy],
            )
            camera.camera_id = i + 1
            reconstruction.add_camera(camera)
            camera_ids.append(i + 1)

    # Convert extrinsics to quaternion + translation
    quaternions, translations = _extrinsic_to_colmap_format(extrinsic)

    # Add images
    for i, img_path in enumerate(image_paths):
        qvec = quaternions[i].astype(float)
        tvec = translations[i].astype(float)

        image = pycolmap.Image(
            id=i + 1,
            name=img_path.name,
            camera_id=camera_ids[i],
            qvec=qvec,
            tvec=tvec,
        )
        reconstruction.add_image(image)

    # Add 3D points with tracks
    for point in points3D:
        point_id = point["id"] + 1
        xyz = point["xyz"].astype(float)
        rgb = point["rgb"].astype(np.uint8)
        track = point["track"]

        # Create pycolmap track
        pycolmap_track = pycolmap.Track()
        for img_idx, point2d_idx in track:
            pycolmap_track.add_element(img_idx + 1, point2d_idx)

        # Add point3D to reconstruction
        point3d = reconstruction.add_point3D(
            xyz=xyz,
            track=pycolmap_track,
            color=rgb,
        )

    # Register 2D points with images
    for img_idx, points_2d in enumerate(image_points2D):
        image = reconstruction.images[img_idx + 1]
        points2D = []

        for x, y, point3d_id in points_2d:
            point2d = pycolmap.Point2D()
            point2d.xy = np.array([float(x), float(y)])
            point2d.point3D_id = point3d_id + 1
            points2D.append(point2d)

        image.points2D = points2D
        image.registered = True

    if verbose:
        CONSOLE.print(f"  - Created reconstruction:")
        CONSOLE.print(f"    - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"    - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"    - Points3D: {len(reconstruction.points3D)}")

    return reconstruction


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


def _extract_from_pycolmap_reconstruction(
    reconstruction,
    image_paths: List[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], List[List]]:
    """Extract refined data from pycolmap Reconstruction after bundle adjustment.

    Returns:
        quaternions: (N, 4) array of quaternions
        translations: (N, 3) array of translations
        intrinsics: (N, 3, 3) array of intrinsic matrices
        points3D: List of 3D point dictionaries
        image_points2D: List of 2D points per image
    """
    import pycolmap

    num_images = len(image_paths)

    # Extract camera poses
    quaternions = []
    translations = []
    for i, img_path in enumerate(image_paths):
        image = reconstruction.images[i + 1]
        quaternions.append(image.qvec)
        translations.append(image.tvec)

    quaternions = np.array(quaternions)
    translations = np.array(translations)

    # Extract intrinsics
    intrinsics = []
    for i in range(num_images):
        image = reconstruction.images[i + 1]
        camera = reconstruction.cameras[image.camera_id]

        # Build intrinsic matrix from camera params
        if camera.model_name == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ])
        elif camera.model_name == "PINHOLE":
            fx, fy, cx, cy = camera.params
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        else:
            # Default to identity
            K = np.eye(3)

        intrinsics.append(K)

    intrinsics = np.array(intrinsics)

    # Extract 3D points
    points3D = []
    for point3d_id, point3d in reconstruction.points3D.items():
        track = []
        for element in point3d.track.elements:
            track.append((element.image_id - 1, element.point2D_idx))

        points3D.append({
            "id": point3d_id - 1,
            "xyz": point3d.xyz,
            "rgb": point3d.color,
            "error": point3d.error,
            "track": track,
        })

    # Extract 2D points per image
    image_points2D = []
    for i in range(num_images):
        image = reconstruction.images[i + 1]
        points_2d = []

        for point_idx, point2d in enumerate(image.points2D):
            if point2d.point3D_id != pycolmap.kInvalidPoint3DId:
                x, y = point2d.xy
                point3d_id = point2d.point3D_id - 1
                points_2d.append((x, y, point3d_id))

        image_points2D.append(points_2d)

    return quaternions, translations, intrinsics, points3D, image_points2D


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


def _extrinsic_to_colmap_format(extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert extrinsic matrices to COLMAP format (quaternion + translation).

    Args:
        extrinsics: Camera extrinsic matrices (N, 4, 4) or (N, 3, 4)

    Returns:
        Tuple of:
        - quaternions: (N, 4) array in COLMAP format [w, x, y, z]
        - translations: (N, 3) array of camera positions
    """
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []

    for i in range(num_cameras):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]

        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        quaternions.append(quat)
        translations.append(t)

    return np.array(quaternions), np.array(translations)


def _filter_and_prepare_points(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    colors_rgb: np.ndarray,
    conf_threshold: float,
    stride: int,
    mask_black_bg: bool,
    mask_white_bg: bool,
    verbose: bool,
) -> Tuple[List[Dict], List[List]]:
    """Filter points based on confidence and prepare for COLMAP format.

    This function is used by run_vggt() for depth-based reconstruction.
    It filters 3D points by confidence and optional color masks, then builds
    tracks by hashing nearby points together.

    Args:
        world_points: 3D points from depth unprojection (S, H, W, 3)
        world_points_conf: Confidence values for each point (S, H, W)
        colors_rgb: RGB colors for each point (S, H, W, 3)
        conf_threshold: Percentile threshold for confidence (0-100)
        stride: Sampling stride for points (higher = fewer points)
        mask_black_bg: If True, filter out very dark points
        mask_white_bg: If True, filter out very bright points
        verbose: If True, log progress

    Returns:
        Tuple of:
        - points3D: List of 3D point dictionaries with tracks
        - image_points2D: List of 2D points per image
    """
    S, H, W = world_points.shape[:3]

    vertices_3d = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb.reshape(-1, 3)

    # Compute confidence threshold
    if conf_threshold == 0.0:
        conf_thres_value = 0.0
    else:
        conf_thres_value = np.percentile(conf, conf_threshold)

    if verbose:
        CONSOLE.print(f"  - Confidence threshold: {conf_threshold}% (value: {conf_thres_value:.4f})")

    conf_mask = (conf >= conf_thres_value) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        white_bg_mask = ~(
            (colors_rgb_flat[:, 0] > 240) &
            (colors_rgb_flat[:, 1] > 240) &
            (colors_rgb_flat[:, 2] > 240)
        )
        conf_mask = conf_mask & white_bg_mask

    # Build 3D points and tracks
    points3D = []
    point_indices = {}
    image_points2D = [[] for _ in range(S)]

    for img_idx in range(S):
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                flat_idx = img_idx * H * W + y * W + x

                if flat_idx >= len(conf):
                    continue

                if not conf_mask[flat_idx]:
                    continue

                point3D = vertices_3d[flat_idx]
                rgb = colors_rgb_flat[flat_idx]

                if not np.all(np.isfinite(point3D)):
                    continue

                point_hash = _hash_point(point3D, scale=100)

                if point_hash not in point_indices:
                    point_idx = len(points3D)
                    point_indices[point_hash] = point_idx

                    point_entry = {
                        "id": point_idx,
                        "xyz": point3D,
                        "rgb": rgb,
                        "error": 1.0,
                        "track": [(img_idx, len(image_points2D[img_idx]))],
                    }
                    points3D.append(point_entry)
                else:
                    point_idx = point_indices[point_hash]
                    points3D[point_idx]["track"].append((img_idx, len(image_points2D[img_idx])))

                image_points2D[img_idx].append((x, y, point_indices[point_hash]))

    if verbose:
        CONSOLE.print(f"  - Prepared {len(points3D)} 3D points")

    return points3D, image_points2D


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


def _write_colmap_cameras_bin(file_path: Path, intrinsics: np.ndarray, image_width: int, image_height: int) -> None:
    """Write camera intrinsics to COLMAP cameras.bin format.

    Writes camera parameters in COLMAP's binary format using the PINHOLE model.

    Args:
        file_path: Output path for cameras.bin file
        intrinsics: Camera intrinsic matrices (N, 3, 3)
        image_width: Image width in pixels
        image_height: Image height in pixels
    """
    with open(file_path, "wb") as fid:
        fid.write(struct.pack("<Q", len(intrinsics)))

        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1
            model_id = 1  # PINHOLE

            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])

            fid.write(struct.pack("<I", camera_id))
            fid.write(struct.pack("<I", model_id))
            fid.write(struct.pack("<Q", image_width))
            fid.write(struct.pack("<Q", image_height))
            fid.write(struct.pack("<dddd", fx, fy, cx, cy))


def _write_colmap_images_bin(
    file_path: Path,
    quaternions: np.ndarray,
    translations: np.ndarray,
    image_points2D: List[List],
    image_paths: List[Path],
) -> None:
    """Write camera poses and keypoints to COLMAP images.bin format.

    Args:
        file_path: Output path for images.bin file
        quaternions: Camera rotations in quaternion format (N, 4) [w, x, y, z]
        translations: Camera translations (N, 3)
        image_points2D: List of 2D points per image, each as [(x, y, point3d_id), ...]
        image_paths: Paths to image files for extracting names
    """
    with open(file_path, "wb") as fid:
        fid.write(struct.pack("<Q", len(quaternions)))

        for i in range(len(quaternions)):
            image_id = i + 1
            camera_id = i + 1

            qw, qx, qy, qz = quaternions[i].astype(float)
            tx, ty, tz = translations[i].astype(float)

            image_name = image_paths[i].name.encode()
            points = image_points2D[i]

            fid.write(struct.pack("<I", image_id))
            fid.write(struct.pack("<dddd", qw, qx, qy, qz))
            fid.write(struct.pack("<ddd", tx, ty, tz))
            fid.write(struct.pack("<I", camera_id))
            fid.write(image_name + b"\x00")

            fid.write(struct.pack("<Q", len(points)))

            for x, y, point3d_id in points:
                fid.write(struct.pack("<ddQ", float(x), float(y), point3d_id + 1))


def _write_colmap_points3D_bin(file_path: Path, points3D: List[Dict]) -> None:
    """Write 3D points and tracks to COLMAP points3D.bin format.

    Args:
        file_path: Output path for points3D.bin file
        points3D: List of 3D point dictionaries with keys:
            - id: Point ID
            - xyz: 3D coordinates (3,)
            - rgb: RGB color (3,) in range [0, 255]
            - error: Reprojection error
            - track: List of (image_id, point2d_idx) tuples
    """
    with open(file_path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))

        for point in points3D:
            point_id = point["id"] + 1
            x, y, z = point["xyz"].astype(float)
            r, g, b = point["rgb"].astype(np.uint8)
            error = float(point["error"])
            track = point["track"]

            fid.write(struct.pack("<Q", point_id))
            fid.write(struct.pack("<ddd", x, y, z))
            fid.write(struct.pack("<BBB", int(r), int(g), int(b)))
            fid.write(struct.pack("<d", error))

            fid.write(struct.pack("<Q", len(track)))
            for img_id, point2d_idx in track:
                fid.write(struct.pack("<II", img_id + 1, point2d_idx))

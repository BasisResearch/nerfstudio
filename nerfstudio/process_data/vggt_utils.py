"""
Code that uses VGGT (Visual Geometry Grounded deep Transformer)
to estimate camera poses and depth maps for structure from motion.
Requires vggt module from : https://github.com/facebookresearch/vggt
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

import os
import sys
import struct
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import CONSOLE


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
    """Runs VGGT on the images to estimate camera poses and depth.

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

    try:
        import torch
        from PIL import Image
    except ImportError:
        _HAS_TORCH = False
    else:
        _HAS_TORCH = True

    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except ImportError:
        _HAS_VGGT = False
    else:
        _HAS_VGGT = True

    if not _HAS_TORCH:
        CONSOLE.print("[bold red]Error: PyTorch is required to use VGGT!")
        sys.exit(1)

    if not _HAS_VGGT:
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

    # Create output directory
    output_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    original_width, original_height = original_images[0].shape[1], original_images[0].shape[0]

    # Load and preprocess images for VGGT
    image_names = [str(p) for p in image_paths]
    images = load_and_preprocess_images(image_names).to(device)

    if verbose:
        CONSOLE.print(f"[bold yellow]Running VGGT inference...")

    # Run inference
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to camera parameters
    extrinsic, intrinsic_downsampled = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], [original_height, original_width]
    )

    # Move predictions to CPU
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    extrinsic = extrinsic.cpu().numpy().squeeze(0)
    intrinsic = intrinsic.cpu().numpy().squeeze(0)
    intrinsic_downsampled = intrinsic_downsampled.cpu().numpy().squeeze(0)

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

    # Convert camera poses to COLMAP format
    quaternions, translations = _extrinsic_to_colmap_format(extrinsic)

    # Filter and prepare 3D points
    depth_conf = predictions.get("depth_conf", np.ones_like(depth_map[..., 0]))

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


def _extrinsic_to_colmap_format(extrinsics: np.ndarray):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []

    for i in range(num_cameras):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]

        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]

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
):
    """Filter points based on confidence and prepare for COLMAP format."""
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


def _hash_point(point: np.ndarray, scale: float = 100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)


def _write_colmap_cameras_bin(file_path: Path, intrinsics: np.ndarray, image_width: int, image_height: int):
    """Write camera intrinsics to COLMAP cameras.bin format."""
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
    image_points2D: list,
    image_paths: list,
):
    """Write camera poses and keypoints to COLMAP images.bin format."""
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


def _write_colmap_points3D_bin(file_path: Path, points3D: list):
    """Write 3D points and tracks to COLMAP points3D.bin format."""
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

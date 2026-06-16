import numpy as np
import open3d as o3d
from glob import glob
from PIL import Image

from calibration.calibration import (
    maxDepth,
    R,
    # t_x, t_y, t_z,
    # K_d
)
from calibration.intrinsics import estimate_intrinsics
from .edge_detection import detect_edges, save_edges
from visualization_open3d.visualization import (
    inside_pointcloud,
    origin_pointcloud
)


def backproject_edges(depth_array, rgb_array, edge_mask, fx, fy, cx, cy):
    """
    Back-projects only edge pixels from a depth map into a coloured 3D point cloud.

    Args:
        depth_array (np.ndarray): Raw depth image (H, W), values in mm.
        rgb_array   (np.ndarray): RGB image (H, W, 3), uint8.
        edge_mask   (np.ndarray): Binary edge mask (H, W), uint8. 255 = edge.
        fx, fy      (float):      Focal lengths (pixels).
        cx, cy      (float):      Principal point (pixels).

    Returns:
        points (np.ndarray): 3D edge point positions (N, 3) in metres.
        colors (np.ndarray): Normalised RGB colours (N, 3), range [0, 1].
    """
    h, w   = depth_array.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_array.astype(np.float32) / 1000.0

    # Keep only pixels that are both valid depth AND on an edge
    edge_pixels = edge_mask > 0
    valid       = (z > 0) & (z <= maxDepth) & edge_pixels

    z, uu, vv = z[valid], uu[valid], vv[valid]

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    colors = rgb_array[valid] / 255.0

    return points, colors


# def flip_pointcloud(points):
#     """
#     Converts from depth camera convention (Y-down) to Open3D convention (Y-up).
#     """
#     points = points.copy()
#     points[:, 1] *= -1

#     R_z = np.array([
#         [0, -1, 0],
#         [1,  0, 0],
#         [0,  0, 1]
#     ])
#     points = (R_z @ points.T).T
#     return points
def flip_pointcloud(points):
    points = points.copy()
    points[:, 1] *= -1  # Y-down to Y-up
    return points


def run_edge_reconstruction():
    """
    Loads an RGB-D pair, detects edges, and reconstructs only edge pixels in 3D.

    Processing steps:
        1. Load RGB and depth images from disk.
        2. Detect edges in the RGB image and save to Edges/.
        3. Estimate camera intrinsics from the RGB image.
        4. Back-project only edge pixels to 3D using depth intrinsics.
        5. Apply extrinsic rotation R to align into RGB camera space.
        6. Correct coordinate convention for Open3D (Y-up).
        7. Visualise the resulting edge point cloud.
    """
    rgb_images   = glob("extracted_dataset/images/*.png")
    depth_images = glob("extracted_dataset/depths/*.png")

    rgb_array   = np.asarray(Image.open(rgb_images[0]).convert("RGB"))
    depth_array = np.asarray(Image.open(depth_images[0]))

    print(f"RGB shape:   {rgb_array.shape}")
    print(f"Depth shape: {depth_array.shape}")

    # Detect edges and save to Edges/ directory
    edge_mask = detect_edges(rgb_array, low_threshold=20, high_threshold=80)
    save_edges(edge_mask, filename="frame_0000_edges.png")

    print(f"Edge pixels: {np.count_nonzero(edge_mask):,} / {edge_mask.size:,}")

    # Estimate intrinsics from RGB image
    intrinsics = estimate_intrinsics(rgb_array)
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Back-project edge pixels only
    points, colors = backproject_edges(
        depth_array,
        rgb_array,
        edge_mask,
        fx, fy, cx, cy
    )

    # Apply extrinsic rotation into RGB camera space
    points = (R @ points.T).T

    # Reorient to Open3D convention (Y-up)
    points = flip_pointcloud(points)

    print(f"Edge point cloud: {len(points):,} points")
    print(f"Depth range: {points[:, 2].min():.3f}m — {points[:, 2].max():.3f}m")

    # Construct and display the Open3D point cloud
    inside_pointcloud(points, colors, axis=True)
    # origin_pointcloud(points, colors, axis=True)



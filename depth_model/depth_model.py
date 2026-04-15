import numpy as np
import cv2
from glob import glob
import open3d as o3d
from PIL import Image
from calibration.calibration import (
    maxDepth,
    fx_rgb, fy_rgb, cx_rgb, cy_rgb,
    k1_rgb, k2_rgb, k3_rgb, p1_rgb, p2_rgb,
    fx_d, fy_d, cx_d, cy_d,
    k1_d, k2_d, k3_d, p1_d, p2_d,
    R,
    t_x, t_y, t_z,
    K_rgb,
    K_d
)

# =============================================================================
# NYU Depth V2 — RGB-D Point Cloud Reconstruction
# =============================================================================
# Calibration parameters sourced from the RGBDemo Calibration Tool.
#
# Pipeline:
#   1. Undistort RGB and depth images using their respective lens models
#   2. Back-project depth pixels to 3D using depth camera intrinsics
#   3. Transform 3D points into RGB camera space via extrinsic (R, t)
#   4. Reconstruct coloured point cloud and visualise with Open3D
# =============================================================================


def make_intrinsics():
    """
    Returns the depth camera intrinsic parameters.

    Returns:
        tuple: (fx_d, fy_d, cx_d, cy_d)
    """
    return fx_d, fy_d, cx_d, cy_d


def backproject(depth_array, rgb_array, fx, fy, cx, cy, K_d):
    """
    Back-projects a depth map into a coloured 3D point cloud.

    Applies lens undistortion to the depth image, converts depth values
    from millimetres to metres, and reprojects each valid pixel into 3D
    using the pinhole camera model.

    Args:
        depth_array (np.ndarray): Raw depth image (H, W), values in mm.
        rgb_array   (np.ndarray): Undistorted RGB image (H, W, 3), uint8.
        fx, fy      (float):      Depth camera focal lengths (pixels).
        cx, cy      (float):      Depth camera principal point (pixels).
        K_d         (np.ndarray): Depth camera intrinsic matrix (3, 3).

    Returns:
        points (np.ndarray): 3D point positions (N, 3) in metres.
        colors (np.ndarray): Normalised RGB colours (N, 3), range [0, 1].
    """
    # Correct lens distortion on the depth image
    dist_d      = np.array([k1_d, k2_d, p1_d, p2_d, k3_d])
    depth_array = cv2.undistort(depth_array.astype(np.float32), K_d, dist_d)

    # Build a pixel coordinate grid covering every pixel in the depth image
    h, w       = depth_array.shape
    uu, vv     = np.meshgrid(np.arange(w), np.arange(h))

    # Convert raw depth values from millimetres to metres
    z = depth_array.astype(np.float32) / 1000.0

    # Mask out invalid readings and points exceeding the depth threshold
    valid = (z > 0) & (z <= maxDepth)
    z, uu, vv = z[valid], uu[valid], vv[valid]

    # Apply inverse pinhole projection: (u, v, z) → (X, Y, Z)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)   # (N, 3)
    colors = rgb_array[valid] / 255.0        # (N, 3) normalised

    # -------------------------------------------------------------------------
    # Potential extra step if have time:
    # Color alignment using p = KP
    # where K is camera intrinsics matrix
    # -------------------------------------------------------------------------

    return points, colors


def flip_pointcloud(points):
    """
    Converts from depth camera convention (Y-down) to Open3D convention (Y-up).

    The depth camera coordinate system uses a downward-facing Y axis, which is
    standard in computer vision. Open3D expects Y-up. Negating Y corrects the
    orientation without affecting the X (left-right) or Z (depth) axes.

    Args:
        points (np.ndarray): Point cloud array (N, 3).

    Returns:
        np.ndarray: Reoriented point cloud (N, 3).
    """
    points = points.copy()
    points[:, 1] *= -1
    return points


def run_depth_model():
    """
    Loads an RGB-D pair, reconstructs a coloured 3D point cloud, and
    launches an interactive Open3D visualisation window.

    Processing steps:
        1. Load RGB and depth images from disk.
        2. Undistort the RGB image using the calibrated lens model.
        3. Back-project the depth image to 3D using depth intrinsics.
        4. Apply the extrinsic transform (R, t) to align points into
           RGB camera space.
        5. Correct coordinate convention for Open3D (Y-up).
        6. Visualise the resulting point cloud.
    """
    rgb_images   = glob("extracted_dataset/images/*.png")
    depth_images = glob("extracted_dataset/depths/*.png")

    rgb_array   = np.asarray(Image.open(rgb_images[0]).convert("RGB"))
    depth_array = np.asarray(Image.open(depth_images[0]))

    print(f"RGB shape:   {rgb_array.shape}")
    print(f"Depth shape: {depth_array.shape}")

    # Correct RGB lens distortion prior to colour sampling
    dist_rgb  = np.array([k1_rgb, k2_rgb, p1_rgb, p2_rgb, k3_rgb])
    rgb_array = cv2.undistort(rgb_array, K_rgb, dist_rgb)

    # Retrieve calibrated depth camera intrinsics
    fx, fy, cx, cy = make_intrinsics()

    # Back-project depth pixels to 3D and sample RGB colours
    points, colors = backproject(depth_array, rgb_array, fx, fy, cx, cy, K_d)

    # Apply extrinsic transform: rotate and translate into RGB camera space
    t      = np.array([t_x, t_y, t_z])
    points = (R @ points.T).T + t

    # Reorient to Open3D coordinate convention (Y-up)
    points = flip_pointcloud(points)

    print(f"Point cloud: {len(points):,} points")
    print(f"Depth range: {points[:, 2].min():.3f}m — {points[:, 2].max():.3f}m")

    # Construct and display the Open3D point cloud
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="RGB-D Point Cloud Reconstruction",
        width=1200,
        height=800,
    )
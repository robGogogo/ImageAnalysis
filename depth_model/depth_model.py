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

def make_intrinsics():
    """Returns the depth camera intrinsic parameters."""
    return fx_d, fy_d, cx_d, cy_d


def backproject(depth_array, fx, fy, cx, cy, K_d):
    """
    Back-projects a depth map into a 3D point cloud.
    
    Uses Nearest-Neighbor interpolation to preserve depth boundaries.
    """
    dist_d = np.array([k1_d, k2_d, p1_d, p2_d, k3_d])
    h, w = depth_array.shape
    
    # Generate maps for undistortion
    map1, map2 = cv2.initUndistortRectifyMap(K_d, dist_d, None, K_d, (w, h), cv2.CV_32FC1)
    
    # Remap depth using Nearest Neighbor to avoid 'interpolated' phantom points
    depth_undistorted = cv2.remap(depth_array.astype(np.float32), map1, map2, interpolation=cv2.INTER_NEAREST)

    # Build a pixel coordinate grid
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))

    # Convert raw depth values from millimetres to metres
    z = depth_undistorted / 1000.0

    # Mask out invalid readings and points exceeding the depth threshold
    valid = (z > 0) & (z <= maxDepth)
    z, uu, vv = z[valid], uu[valid], vv[valid]

    # Apply inverse pinhole projection: (u, v, z) → (X, Y, Z)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)   # (N, 3)
    
    return points


def flip_pointcloud(points):
    """
    Converts from OpenCV (Y-down) to Open3D-friendly (Y-up) view.
    We only flip Y to avoid mirroring the Z-axis (which makes it look inside-out).
    """
    points = points.copy()
    points[:, 1] *= -1  
    return points


def run_depth_model():
    """
    Full pipeline: Load -> Undistort -> Back-project -> Transform -> Align Color -> Visualize
    """
    # 1. Load Data (Sorted to ensure RGB and Depth frames match)
    rgb_images   = sorted(glob("extracted_dataset/images/*.png"))
    depth_images = sorted(glob("extracted_dataset/depths/*.png"))

    if not rgb_images or not depth_images:
        print("Error: No images found in the specified directories.")
        return

    rgb_array   = np.asarray(Image.open(rgb_images[0]).convert("RGB"))
    depth_array = np.asarray(Image.open(depth_images[0]))

    print(f"Processing frame: {rgb_images[0]}")

    # 2. Correct RGB lens distortion
    dist_rgb  = np.array([k1_rgb, k2_rgb, p1_rgb, p2_rgb, k3_rgb])
    rgb_undistorted = cv2.undistort(rgb_array, K_rgb, dist_rgb)

    # 3. Back-project depth pixels to 3D (meters)
    fx, fy, cx, cy = make_intrinsics()
    points_depth_space = backproject(depth_array, fx, fy, cx, cy, K_d)

    # 4. Apply Extrinsic Transform
    # IMPORTANT: Calibration tool 't' is usually in mm, but points are in meters.
    t = np.array([t_x, t_y, t_z]) / 1000.0 
    points_rgb_space = (R @ points_depth_space.T).T + t

    # 5. Color Alignment (Project 3D points onto the 2D RGB image plane)
    u_rgb = (points_rgb_space[:, 0] * fx_rgb / points_rgb_space[:, 2]) + cx_rgb
    v_rgb = (points_rgb_space[:, 1] * fy_rgb / points_rgb_space[:, 2]) + cy_rgb

    # Round to nearest pixel
    u_rgb = np.round(u_rgb).astype(int)
    v_rgb = np.round(v_rgb).astype(int)

    # Filter points that fall outside the RGB camera's field of view
    h_rgb, w_rgb = rgb_undistorted.shape[:2]
    valid_mask = (u_rgb >= 0) & (u_rgb < w_rgb) & (v_rgb >= 0) & (v_rgb < h_rgb)

    final_points = points_rgb_space[valid_mask]
    final_colors = rgb_undistorted[v_rgb[valid_mask], u_rgb[valid_mask]] / 255.0

    # 6. Reorient for Visualization
    final_points = flip_pointcloud(final_points)

    print(f"Point cloud: {len(final_points):,} points")

    # 7. Construct and display the Open3D point cloud
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="RGB-D Point Cloud Reconstruction",
        width=1200,
        height=800,
    )

if __name__ == "__main__":
    run_depth_model()

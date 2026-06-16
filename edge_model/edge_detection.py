import os
import numpy as np
import cv2
from PIL import Image


def detect_edges(rgb_array: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detects edges in an RGB image using the Canny algorithm.

    Args:
        rgb_array      (np.ndarray): Input RGB image (H, W, 3), uint8.
        low_threshold  (int):        Lower hysteresis threshold for Canny.
        high_threshold (int):        Upper hysteresis threshold for Canny.

    Returns:
        np.ndarray: Binary edge mask (H, W), uint8. 255 = edge, 0 = non-edge.
    """
    gray  = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def save_edges(edges: np.ndarray, filename: str, output_dir: str = "Edges") -> str:
    """
    Saves an edge mask image to the Edges/ directory.

    Args:
        edges      (np.ndarray): Binary edge mask (H, W), uint8.
        filename   (str):        Output filename, e.g. "frame_0001_edges.png".
        output_dir (str):        Directory to save into. Created if missing.

    Returns:
        str: Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    Image.fromarray(edges).save(out_path)
    print(f"Saved edge image: {out_path}")
    return out_path


def load_edges(path: str) -> np.ndarray:
    """
    Loads an edge mask from disk.

    Args:
        path (str): Path to a saved edge image.

    Returns:
        np.ndarray: Binary edge mask (H, W), uint8.
    """
    return np.asarray(Image.open(path).convert("L"))
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "geocalib"))

import torch
import numpy as np
from geocalib import GeoCalib

_model = None


def _get_geocalib():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = GeoCalib().to(device)
        _model.eval()
    return _model


def estimate_intrinsics(rgb_array: np.ndarray) -> dict:
    """
    Estimates camera intrinsics from a single RGB image using GeoCalib.

    Args:
        rgb_array: (H, W, 3) uint8 numpy array

    Returns:
        dict with keys: fx, fy, cx, cy, K
    """
    model  = _get_geocalib()
    device = next(model.parameters()).device

    img_tensor = (
        torch.from_numpy(rgb_array.copy())
             .float()
             .permute(2, 0, 1)
             .div(255.0)
             .to(device)
    )

    with torch.no_grad():
        result = model.calibrate(img_tensor, camera_model="pinhole")

    cam = result["camera"]

    return {
        "fx": cam.f[0][0].item(),
        "fy": cam.f[0][1].item(),
        "cx": cam.c[0][0].item(),
        "cy": cam.c[0][1].item(),
        "K":  cam.K[0].cpu().numpy(),
    }
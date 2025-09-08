"""Cálculo de error de reproyección por frame."""
import numpy as np
from typing import Dict, Tuple
from camera import Camera


def reprojection_error(points_3d: np.ndarray, cameras: Dict[str, Camera], frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    errors = {}
    for cid, cam in cameras.items():
        coords, conf = frame_keypoints[cid]
        valid3d = ~np.isnan(points_3d[:, 0]) & (conf > 0)
        if not np.any(valid3d):
            errors[cid] = float('nan')
            continue
        proj = cam.project(points_3d[valid3d])
        diff = proj - coords[valid3d]
        err = np.linalg.norm(diff, axis=1)
        errors[cid] = float(np.mean(err))
    return errors
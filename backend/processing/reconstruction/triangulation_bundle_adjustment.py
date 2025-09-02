"""Refinamiento por Bundle Adjustment (optimiza sólo puntos 3D por frame)."""
import numpy as np
from typing import Dict, Tuple
from scipy.optimize import least_squares
from .camera import Camera


def _residuals(points_3d_flat: np.ndarray, cameras: Dict[str, Camera], frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]], valid_mask: np.ndarray) -> np.ndarray:
    K = valid_mask.shape[0]
    pts3d = points_3d_flat.reshape(K, 3)
    res = []
    for cid, cam in cameras.items():
        coords, conf = frame_keypoints[cid]
        valid = valid_mask & (conf > 0)
        if not np.any(valid):
            continue
        proj = cam.project(pts3d[valid])
        diff = (proj - coords[valid])  # (N,2)
        res.append(diff.reshape(-1))
    if not res:
        return np.zeros(0)
    return np.concatenate(res)


def refine_frame_bundle_adjustment(initial_points_3d: np.ndarray, cameras: Dict[str, Camera], frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    pts3d = initial_points_3d.copy()
    valid_mask = ~np.isnan(pts3d[:, 0])
    if not np.any(valid_mask):
        return pts3d
    x0 = pts3d.reshape(-1)
    result = least_squares(_residuals, x0, args=(cameras, frame_keypoints, valid_mask), method="lm", max_nfev=50)
    refined = result.x.reshape(-1, 3)
    # Mantener NaN en puntos no triangulados
    refined[~valid_mask] = np.nan
    return refined
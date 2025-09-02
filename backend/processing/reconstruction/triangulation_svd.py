"""Triangulación por DLT (SVD) punto a punto para cada frame."""
import numpy as np
from typing import Dict, Tuple
from .camera import Camera


def triangulate_frame_svd(cameras: Dict[str, Camera], frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]], confidence_threshold: float = 0.0) -> np.ndarray:
    """Triangula todos los keypoints de un frame.

    frame_keypoints: dict camera_id -> (coords(K,2), conf(K,))
    Devuelve (K,3) con np.nan donde no se pudo triangulizar.
    """
    cam_items = sorted(cameras.items())
    num_keypoints = next(iter(frame_keypoints.values()))[0].shape[0]
    points_3d = np.full((num_keypoints, 3), np.nan, dtype=np.float64)
    # Precompute projection matrices
    proj = {cid: cam.P for cid, cam in cam_items}
    for k in range(num_keypoints):
        obs = []  # list of (P, x, y)
        for cid, cam in cam_items:
            coords, conf = frame_keypoints[cid]
            if conf[k] > confidence_threshold and not np.isnan(coords[k]).any():
                x, y = coords[k]
                obs.append((proj[cid], x, y))
        if len(obs) < 2:
            continue
        A_rows = []
        for P, x, y in obs:
            A_rows.append(x * P[2] - P[0])
            A_rows.append(y * P[2] - P[1])
        A = np.stack(A_rows)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        points_3d[k] = X[:3]
    return points_3d
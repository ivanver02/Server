"""Clase Camera: gestiona parámetros intrínsecos (desde config) y extrínsecos.

Extrínsecos siempre relativos a camera0 (referencia):
camera0 -> R = I, t = 0
Otras cámaras -> R, t calculados en runtime para cada sesión usando sólo keypoints 2D procesados.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional
from pathlib import Path
import sys

# Asegura que la raíz del proyecto (Server/) esté en sys.path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from config.camera_intrinsics import CAMERA_INTRINSICS


@dataclass
class Camera:
    camera_id: str  # e.g. "camera0"
    K: np.ndarray
    dist_coeffs: np.ndarray
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3x1

    @staticmethod
    def create(camera_id: str) -> "Camera":
        intr = CAMERA_INTRINSICS[camera_id]
        return Camera(
            camera_id=camera_id,
            K=intr["camera_matrix"].astype(np.float64),
            dist_coeffs=intr["distortion_coeffs"].astype(np.float64),
            R=np.eye(3, dtype=np.float64),
            t=np.zeros((3, 1), dtype=np.float64),
        )

    @property
    def P(self) -> np.ndarray:
        """Matriz de proyección 3x4 = K [R|t]."""
        return self.K @ np.hstack([self.R, self.t])

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Proyecta puntos 3D (N,3) -> (N,2) en píxeles."""
        if points_3d.size == 0:
            return np.zeros((0, 2))
        pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (N,4)
        proj = (self.P @ pts_h.T).T  # (N,3)
        proj[:, 0] /= proj[:, 2]
        proj[:, 1] /= proj[:, 2]
        return proj[:, :2]
"""
Sistema de reconstrucción 3D - Clase Camera y gestión de parámetros.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional

from config.camera_intrinsics import CAMERA_INTRINSICS


class Camera:
    """
    Clase Camera que gestiona parámetros intrínsecos y extrínsecos.
    """
    
    def __init__(self, camera_id: str):
        """
        Inicializa la cámara con parámetros intrínsecos del config.
        
        Args:
            camera_id: ID de la cámara ("camera0", "camera1", "camera2")
        """
        if camera_id not in CAMERA_INTRINSICS:
            raise ValueError(f"Camera ID {camera_id} no encontrado en CAMERA_INTRINSICS")
            
        self.camera_id = camera_id
        intrinsics = CAMERA_INTRINSICS[camera_id]
        
        # Parámetros intrínsecos
        self.K = intrinsics["camera_matrix"].copy()
        self.dist_coeffs = intrinsics["distortion_coeffs"].copy()
        self.resolution = intrinsics["resolution"]
        
        # Parámetros extrínsecos (se calculan después)
        self.R = np.eye(3, dtype=np.float64)  # Rotación
        self.t = np.zeros((3, 1), dtype=np.float64)  # Traslación
    
    def set_extrinsics(self, R: np.ndarray, t: np.ndarray):
        """Establece los parámetros extrínsecos."""
        self.R = R.astype(np.float64)
        self.t = t.reshape(3, 1).astype(np.float64)
    
    def get_projection_matrix(self) -> np.ndarray:
        """Retorna la matriz de proyección P = K[R|t]."""
        Rt = np.hstack([self.R, self.t])
        return self.K @ Rt
    
    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Proyecta puntos 3D a coordenadas 2D.
        
        Args:
            points_3d: Array Nx3 con puntos 3D
            
        Returns:
            Array Nx2 con coordenadas 2D
        """
        if len(points_3d) == 0:
            return np.array([]).reshape(0, 2)
            
        # Transformar a coordenadas de cámara
        points_cam = self.R @ points_3d.T + self.t
        
        # Proyectar a imagen
        points_2d = self.K @ points_cam
        
        # Normalizar coordenadas homogéneas
        points_2d = points_2d[:2] / points_2d[2]
        
        return points_2d.T

"""
Triangulación 3D usando Bundle Adjustment (método preciso).
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Optional
import logging
from camera import Camera
from triangulation_svd import triangulate_svd

logger = logging.getLogger(__name__)


def triangulate_bundle_adjustment(
    cameras: Dict[str, Camera],
    keypoints_2d: Dict[str, np.ndarray],
    initial_points_3d: Optional[np.ndarray] = None,
    max_iterations: int = 100
) -> np.ndarray:
    """
    Triangulación 3D usando Bundle Adjustment.
    
    Args:
        cameras: Dict con objetos Camera configurados
        keypoints_2d: Dict con keypoints 2D por cámara
        initial_points_3d: Puntos 3D iniciales (si None, usa SVD)
        max_iterations: Máximo número de iteraciones
        
    Returns:
        Array Nx3 con puntos 3D optimizados
    """
    
    # Usar SVD como inicialización si no se proporciona
    if initial_points_3d is None:
        initial_points_3d = triangulate_svd(cameras, keypoints_2d)
    
    if len(initial_points_3d) == 0:
        return np.array([]).reshape(0, 3)
    
    # Preparar observaciones
    observations = []
    observation_cameras = []
    observation_point_indices = []
    
    camera_list = list(cameras.keys())
    
    for point_idx, point_3d in enumerate(initial_points_3d):
        if np.any(np.isnan(point_3d)):
            continue
            
        for camera_id in camera_list:
            if camera_id not in keypoints_2d:
                continue
                
            kp_2d = keypoints_2d[camera_id]
            if point_idx >= len(kp_2d):
                continue
                
            point_2d = kp_2d[point_idx]
            if np.any(np.isnan(point_2d)):
                continue
            
            observations.append(point_2d)
            observation_cameras.append(cameras[camera_id])
            observation_point_indices.append(point_idx)
    
    if len(observations) == 0:
        return initial_points_3d
    
    observations = np.array(observations)
    
    # Filtrar puntos válidos para optimización
    valid_mask = ~np.any(np.isnan(initial_points_3d), axis=1)
    valid_points = initial_points_3d[valid_mask]
    
    if len(valid_points) == 0:
        return initial_points_3d
    
    def residual_function(params):
        """Calcula residuos de reproyección."""
        points_3d = params.reshape(-1, 3)
        residuals = []
        
        for obs, camera, point_idx in zip(observations, observation_cameras, observation_point_indices):
            # Encontrar índice en el array de puntos válidos
            valid_indices = np.where(valid_mask)[0]
            if point_idx not in valid_indices:
                continue
                
            local_idx = np.where(valid_indices == point_idx)[0]
            if len(local_idx) == 0:
                continue
                
            local_idx = local_idx[0]
            if local_idx >= len(points_3d):
                continue
            
            point_3d = points_3d[local_idx]
            projected = camera.project_points(point_3d.reshape(1, 3))[0]
            residual = projected - obs
            residuals.extend(residual)
        
        return np.array(residuals)
    
    try:
        # Optimización
        x0 = valid_points.flatten()
        result = least_squares(
            residual_function,
            x0,
            method='lm',
            max_nfev=max_iterations * len(x0)
        )
        
        optimized_valid_points = result.x.reshape(-1, 3)
        
        # Reconstruir array completo
        optimized_points = initial_points_3d.copy()
        optimized_points[valid_mask] = optimized_valid_points
        
        # Calcular mejora
        initial_cost = np.sum(residual_function(x0)**2)
        final_cost = result.cost * 2
        
        logger.info(f"Bundle Adjustment: costo {initial_cost:.2e} → {final_cost:.2e}")
        
        return optimized_points
        
    except Exception as e:
        logger.error(f"Error en Bundle Adjustment: {e}")
        return initial_points_3d

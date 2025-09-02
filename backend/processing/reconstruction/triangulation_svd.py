"""
Triangulación 3D usando SVD (método rápido).
"""

import numpy as np
from typing import Dict, Tuple
import logging
from backend.processing.reconstruction.camera import Camera

logger = logging.getLogger(__name__)


def triangulate_svd(
    cameras: Dict[str, Camera],
    keypoints_2d: Dict[str, np.ndarray],
    min_cameras: int = 2
) -> np.ndarray:
    """
    Triangulación 3D usando SVD.
    
    Args:
        cameras: Dict con objetos Camera configurados
        keypoints_2d: Dict con keypoints 2D por cámara
        min_cameras: Número mínimo de cámaras para triangular
        
    Returns:
        Array Nx3 con puntos 3D reconstruidos
    """
    
    # Determinar número de keypoints
    num_keypoints = max(len(kp) for kp in keypoints_2d.values()) if keypoints_2d else 0
    
    if num_keypoints == 0:
        return np.array([]).reshape(0, 3)
    
    points_3d = []
    
    for point_idx in range(num_keypoints):
        # Recopilar observaciones válidas
        A_rows = []
        
        for camera_id, camera in cameras.items():
            if camera_id not in keypoints_2d:
                continue
                
            kp_2d = keypoints_2d[camera_id]
            if point_idx >= len(kp_2d):
                continue
                
            point_2d = kp_2d[point_idx]
            if np.any(np.isnan(point_2d)):
                continue
            
            # Matriz de proyección
            P = camera.get_projection_matrix()
            x, y = point_2d
            
            # Ecuaciones de DLT: x*P3 - P1 = 0, y*P3 - P2 = 0
            A_rows.append(x * P[2, :] - P[0, :])
            A_rows.append(y * P[2, :] - P[1, :])
        
        if len(A_rows) < min_cameras * 2:
            points_3d.append([np.nan, np.nan, np.nan])
            continue
        
        A = np.array(A_rows)
        
        try:
            # Resolver usando SVD
            _, _, Vt = np.linalg.svd(A)
            X_hom = Vt[-1, :]
            
            # Convertir a coordenadas cartesianas
            if abs(X_hom[3]) < 1e-10:
                points_3d.append([np.nan, np.nan, np.nan])
                continue
                
            X_cart = X_hom[:3] / X_hom[3]
            
            # Verificar rango razonable
            if np.any(np.abs(X_cart) > 1000):
                points_3d.append([np.nan, np.nan, np.nan])
                continue
                
            points_3d.append(X_cart)
            
        except np.linalg.LinAlgError:
            points_3d.append([np.nan, np.nan, np.nan])
    
    result = np.array(points_3d)
    valid_count = np.sum(~np.isnan(result[:, 0]))
    logger.info(f"Triangulación SVD: {valid_count}/{num_keypoints} puntos válidos")
    
    # Log estadísticas de los puntos 3D
    if valid_count > 0:
        valid_mask = ~np.isnan(result[:, 0])
        valid_points = result[valid_mask]
        logger.info(f"  Rango X: [{valid_points[:, 0].min():.2f}, {valid_points[:, 0].max():.2f}]")
        logger.info(f"  Rango Y: [{valid_points[:, 1].min():.2f}, {valid_points[:, 1].max():.2f}]")
        logger.info(f"  Rango Z: [{valid_points[:, 2].min():.2f}, {valid_points[:, 2].max():.2f}]")
        logger.info(f"  Distancia media al origen: {np.linalg.norm(valid_points, axis=1).mean():.2f}")
    
    return result

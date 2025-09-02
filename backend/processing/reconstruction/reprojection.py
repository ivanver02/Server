"""
Reproyección de puntos 3D a 2D para validación.
"""

import numpy as np
from typing import Dict, Tuple
import logging
from backend.processing.reconstruction.camera import Camera

logger = logging.getLogger(__name__)


def reproject_and_validate(
    cameras: Dict[str, Camera],
    points_3d: np.ndarray,
    keypoints_2d_original: Dict[str, np.ndarray]
) -> Dict[str, Dict]:
    """
    Reproyecta puntos 3D y calcula errores de reproyección.
    
    Args:
        cameras: Dict con objetos Camera configurados
        points_3d: Array Nx3 con puntos 3D reconstruidos
        keypoints_2d_original: Dict con keypoints 2D originales
        
    Returns:
        Dict con estadísticas de reproyección por cámara
    """
    
    validation_results = {}
    
    for camera_id, camera in cameras.items():
        if camera_id not in keypoints_2d_original:
            continue
            
        original_2d = keypoints_2d_original[camera_id]
        
        # Verificar dimensiones compatibles
        num_points = min(len(points_3d), len(original_2d))
        if num_points == 0:
            continue
        
        points_3d_subset = points_3d[:num_points]
        original_2d_subset = original_2d[:num_points]
        
        # Filtrar puntos válidos (sin NaN)
        valid_3d = ~np.any(np.isnan(points_3d_subset), axis=1)
        valid_2d = ~np.any(np.isnan(original_2d_subset), axis=1)
        valid_mask = valid_3d & valid_2d
        
        if np.sum(valid_mask) == 0:
            validation_results[camera_id] = {
                "num_valid": 0,
                "mean_error": np.nan,
                "errors": []
            }
            continue
        
        valid_points_3d = points_3d_subset[valid_mask]
        valid_original_2d = original_2d_subset[valid_mask]
        
        # Reproyectar puntos 3D
        reprojected_2d = camera.project_points(valid_points_3d)
        
        # Calcular errores de reproyección
        errors = np.linalg.norm(reprojected_2d - valid_original_2d, axis=1)
        
        validation_results[camera_id] = {
            "num_valid": len(errors),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "errors": errors.tolist()
        }
        
        logger.info(f"Cámara {camera_id}: Error medio reproyección = {np.mean(errors):.2f} px")
    
    # Estadísticas globales
    all_errors = []
    total_valid = 0
    
    for result in validation_results.values():
        if result["num_valid"] > 0:
            all_errors.extend(result["errors"])
            total_valid += result["num_valid"]
    
    if all_errors:
        validation_results["global"] = {
            "total_valid": total_valid,
            "mean_error": float(np.mean(all_errors)),
            "std_error": float(np.std(all_errors)),
            "max_error": float(np.max(all_errors))
        }
        
        logger.info(f"Error global medio: {np.mean(all_errors):.2f} px ({total_valid} puntos)")
    else:
        validation_results["global"] = {
            "total_valid": 0,
            "mean_error": np.nan
        }
    
    return validation_results

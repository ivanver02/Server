"""
Métodos de triangulación 3D: SVD (rápido) y Bundle Adjustment (preciso).
Implementación minimalista y optimizada para máxima precisión.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import least_squares
from .camera import Camera

logger = logging.getLogger(__name__)


def triangulate_svd(
    cameras: Dict[str, Camera],
    keypoints_2d: Dict[str, np.ndarray],
    min_cameras: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulación 3D usando SVD (rápida, menor precisión).
    
    Usa el método DLT (Direct Linear Transform) para resolver el sistema
    sobredeterminado AX = 0 donde X son las coordenadas 3D homogéneas.
    
    Args:
        cameras: Dict con objetos Camera configurados con extrínsecos
        keypoints_2d: Dict con keypoints 2D por cámara (shape: Nx2)
        min_cameras: Número mínimo de cámaras para triangular un punto
        
    Returns:
        Tupla (points_3d, confidence):
        - points_3d: Array Nx3 con coordenadas 3D
        - confidence: Array N con confianza por punto (número de cámaras usadas)
    """
    
    # Determinar número de keypoints
    num_keypoints = 0
    for kp_2d in keypoints_2d.values():
        num_keypoints = max(num_keypoints, len(kp_2d))
    
    if num_keypoints == 0:
        return np.array([]).reshape(0, 3), np.array([])
    
    points_3d = np.full((num_keypoints, 3), np.nan, dtype=np.float64)
    confidence = np.zeros(num_keypoints, dtype=np.float64)
    
    for point_idx in range(num_keypoints):
        # Recopilar observaciones válidas para este keypoint
        projections = []
        cameras_used = []
        
        for camera_id, camera in cameras.items():
            if camera_id not in keypoints_2d:
                continue
                
            kp_2d = keypoints_2d[camera_id]
            if point_idx >= len(kp_2d):
                continue
                
            point_2d = kp_2d[point_idx]
            
            # Verificar que el punto sea válido (no NaN)
            if np.any(np.isnan(point_2d)):
                continue
                
            projections.append((camera.projection_matrix, point_2d))
            cameras_used.append(camera_id)
        
        # Verificar que tenemos suficientes cámaras
        if len(projections) < min_cameras:
            continue
        
        # Construir sistema AX = 0
        A = []
        for P, (x, y) in projections:
            # Ecuaciones de DLT
            A.append(x * P[2, :] - P[0, :])  # x*P3 - P1 = 0
            A.append(y * P[2, :] - P[1, :])  # y*P3 - P2 = 0
        
        A = np.array(A)
        
        # Resolver usando SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            X_hom = Vt[-1, :]  # Última fila de V^T (última columna de V)
            
            # Convertir de coordenadas homogéneas a cartesianas
            if abs(X_hom[3]) < 1e-10:
                continue  # Punto en el infinito
                
            X_cart = X_hom[:3] / X_hom[3]
            
            # Verificar que el punto esté dentro de un rango razonable
            if np.any(np.abs(X_cart) > 1000):  # 1000 unidades máximo
                continue
                
            points_3d[point_idx] = X_cart
            confidence[point_idx] = len(projections)
            
        except np.linalg.LinAlgError:
            logger.warning(f"Error SVD en keypoint {point_idx}")
            continue
    
    # Filtrar puntos válidos
    valid_mask = ~np.any(np.isnan(points_3d), axis=1)
    points_3d_valid = points_3d[valid_mask]
    confidence_valid = confidence[valid_mask]
    
    logger.info(f"Triangulación SVD: {len(points_3d_valid)}/{num_keypoints} puntos válidos")
    
    return points_3d_valid, confidence_valid


def triangulate_bundle_adjustment(
    cameras: Dict[str, Camera],
    keypoints_2d: Dict[str, np.ndarray],
    initial_points_3d: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    reprojection_threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Triangulación 3D usando Bundle Adjustment (lenta, máxima precisión).
    
    Optimiza simultáneamente las posiciones 3D para minimizar el error de reproyección
    total en todas las cámaras. Usa initial_points_3d como punto de partida.
    
    Args:
        cameras: Dict con objetos Camera configurados con extrínsecos
        keypoints_2d: Dict con keypoints 2D por cámara (shape: Nx2)
        initial_points_3d: Puntos 3D iniciales (ej. desde triangulate_svd)
        max_iterations: Máximo número de iteraciones de optimización
        reprojection_threshold: Umbral para considerar un punto como outlier
        
    Returns:
        Tupla (points_3d, confidence, info):
        - points_3d: Array Nx3 con coordenadas 3D optimizadas
        - confidence: Array N con confianza por punto
        - info: Dict con información de la optimización
    """
    
    # Si no hay puntos iniciales, usar SVD
    if initial_points_3d is None:
        initial_points_3d, _ = triangulate_svd(cameras, keypoints_2d)
    
    if len(initial_points_3d) == 0:
        return np.array([]).reshape(0, 3), np.array([]), {"status": "no_points"}
    
    # Preparar datos para optimización
    observations = []
    camera_list = []
    point_indices = []
    
    camera_ids = list(cameras.keys())
    for point_idx in range(len(initial_points_3d)):
        for cam_idx, camera_id in enumerate(camera_ids):
            if camera_id not in keypoints_2d:
                continue
                
            kp_2d = keypoints_2d[camera_id]
            if point_idx >= len(kp_2d):
                continue
                
            point_2d = kp_2d[point_idx]
            if np.any(np.isnan(point_2d)):
                continue
                
            observations.append(point_2d)
            camera_list.append(cameras[camera_id])
            point_indices.append(point_idx)
    
    if len(observations) == 0:
        return initial_points_3d, np.ones(len(initial_points_3d)), {"status": "no_observations"}
    
    observations = np.array(observations)
    
    # Definir función de residuos
    def residual_function(params):
        """
        Calcula residuos de reproyección para optimización.
        params: array 1D con coordenadas 3D concatenadas [x1,y1,z1,x2,y2,z2,...]
        """
        points_3d = params.reshape(-1, 3)
        residuals = []
        
        for obs_idx, (observation, camera, point_idx) in enumerate(zip(observations, camera_list, point_indices)):
            if point_idx >= len(points_3d):
                continue
                
            point_3d = points_3d[point_idx:point_idx+1]  # Shape (1, 3)
            projected = camera.project_points(point_3d)[0]  # Shape (2,)
            
            residual = projected - observation
            residuals.extend(residual)
        
        return np.array(residuals)
    
    # Optimización con Levenberg-Marquardt
    try:
        x0 = initial_points_3d.flatten()
        
        result = least_squares(
            residual_function,
            x0,
            method='lm',
            max_nfev=max_iterations * len(x0),
            ftol=1e-8,
            xtol=1e-8
        )
        
        optimized_points = result.x.reshape(-1, 3)
        
        # Calcular errores de reproyección finales para confianza
        final_residuals = residual_function(result.x)
        residuals_per_point = final_residuals.reshape(-1, 2)
        reprojection_errors = np.linalg.norm(residuals_per_point, axis=1)
        
        # Agrupar errores por punto (puede haber múltiples observaciones por punto)
        point_errors = {}
        for obs_idx, point_idx in enumerate(point_indices):
            if point_idx not in point_errors:
                point_errors[point_idx] = []
            point_errors[point_idx].append(reprojection_errors[obs_idx])
        
        # Calcular confianza basada en error promedio y número de observaciones
        confidence = np.zeros(len(optimized_points))
        for point_idx in range(len(optimized_points)):
            if point_idx in point_errors:
                avg_error = np.mean(point_errors[point_idx])
                num_observations = len(point_errors[point_idx])
                
                # Confianza inversamente proporcional al error, ponderada por observaciones
                confidence[point_idx] = num_observations / (1.0 + avg_error)
            else:
                confidence[point_idx] = 0.0
        
        # Filtrar outliers
        valid_mask = np.array([
            point_idx in point_errors and np.mean(point_errors[point_idx]) < reprojection_threshold
            for point_idx in range(len(optimized_points))
        ])
        
        optimized_points_filtered = optimized_points[valid_mask]
        confidence_filtered = confidence[valid_mask]
        
        info = {
            "status": "success",
            "iterations": result.nfev,
            "cost_initial": np.sum(residual_function(x0)**2),
            "cost_final": result.cost * 2,  # least_squares usa cost = 0.5 * sum(residuals^2)
            "points_optimized": len(optimized_points),
            "points_valid": len(optimized_points_filtered),
            "outliers_removed": np.sum(~valid_mask)
        }
        
        logger.info(f"Bundle Adjustment: {info['points_valid']}/{info['points_optimized']} puntos válidos, "
                   f"costo: {info['cost_initial']:.2e} → {info['cost_final']:.2e}")
        
        return optimized_points_filtered, confidence_filtered, info
        
    except Exception as e:
        logger.error(f"Error en Bundle Adjustment: {e}")
        
        # Fallback: retornar puntos iniciales con confianza básica
        confidence = np.ones(len(initial_points_3d))
        info = {"status": "failed", "error": str(e)}
        
        return initial_points_3d, confidence, info


def compare_triangulation_methods(
    cameras: Dict[str, Camera],
    keypoints_2d: Dict[str, np.ndarray]
) -> Dict:
    """
    Compara ambos métodos de triangulación y retorna estadísticas.
    
    Args:
        cameras: Dict con objetos Camera configurados
        keypoints_2d: Dict con keypoints 2D por cámara
        
    Returns:
        Dict con resultados y estadísticas de comparación
    """
    
    # Triangulación SVD
    logger.info("Ejecutando triangulación SVD...")
    points_svd, conf_svd = triangulate_svd(cameras, keypoints_2d)
    
    # Bundle Adjustment usando SVD como inicialización
    logger.info("Ejecutando Bundle Adjustment...")
    points_ba, conf_ba, info_ba = triangulate_bundle_adjustment(
        cameras, keypoints_2d, points_svd
    )
    
    # Comparar resultados si ambos tienen puntos
    comparison = {
        "svd": {
            "points": points_svd,
            "confidence": conf_svd,
            "num_points": len(points_svd)
        },
        "bundle_adjustment": {
            "points": points_ba,
            "confidence": conf_ba,
            "num_points": len(points_ba),
            "optimization_info": info_ba
        }
    }
    
    if len(points_svd) > 0 and len(points_ba) > 0:
        # Calcular diferencias entre métodos (para puntos en común)
        min_points = min(len(points_svd), len(points_ba))
        diff_points = points_svd[:min_points] - points_ba[:min_points]
        
        comparison["comparison"] = {
            "mean_difference": np.mean(np.linalg.norm(diff_points, axis=1)),
            "max_difference": np.max(np.linalg.norm(diff_points, axis=1)),
            "points_compared": min_points
        }
        
        logger.info(f"Diferencia promedio SVD vs BA: "
                   f"{comparison['comparison']['mean_difference']:.3f} unidades")
    
    return comparison

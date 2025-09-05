"""
Bundle Adjustment completo que optimiza tanto puntos 3D como parámetros extrínsecos de cámaras.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convierte vector de Rodrigues a matriz de rotación."""
    angle = np.linalg.norm(rvec)
    if angle == 0:
        return np.eye(3)
    
    axis = rvec / angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Fórmula de Rodrigues
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
    return R

def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convierte matriz de rotación a vector de Rodrigues."""
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    if angle == 0:
        return np.zeros(3)
    
    if np.abs(angle - np.pi) < 1e-6:
        # Caso especial cuando angle ≈ π
        # Encontrar el eigenvector correspondiente al eigenvalor 1
        eigenvals, eigenvecs = np.linalg.eigh(R + np.eye(3))
        axis = eigenvecs[:, np.argmax(eigenvals)]
        return angle * axis / np.linalg.norm(axis)
    
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2 * np.sin(angle))
    
    return angle * axis

def project_point(point_3d: np.ndarray, camera) -> np.ndarray:
    """Proyecta un punto 3D a la imagen usando los parámetros de la cámara."""
    # Transformar al sistema de coordenadas de la cámara
    if hasattr(camera, 'R') and hasattr(camera, 't'):
        # Aplicar rotación y traslación
        point_cam = camera.R @ point_3d + camera.t.flatten()
    else:
        point_cam = point_3d
    
    # Proyección perspectiva
    if point_cam[2] <= 0:
        return np.array([np.inf, np.inf])
    
    x_norm = point_cam[0] / point_cam[2]
    y_norm = point_cam[1] / point_cam[2]
    
    # Aplicar parámetros intrínsecos
    u = camera.K[0, 0] * x_norm + camera.K[0, 2]
    v = camera.K[1, 1] * y_norm + camera.K[1, 2]
    
    return np.array([u, v])

def bundle_adjustment_residual(params: np.ndarray, cameras: Dict, 
                              frame_keypoints: Dict, point_indices: np.ndarray,
                              camera_param_sizes: Dict) -> np.ndarray:
    """Función de residuos para Bundle Adjustment completo."""
    
    residuals = []
    param_offset = 0
    
    # Extraer puntos 3D
    n_points = len(point_indices)
    points_3d = params[:n_points * 3].reshape(n_points, 3)
    param_offset += n_points * 3
    
    # Extraer parámetros de cámaras y actualizar cámaras
    updated_cameras = {}
    for cam_id in ["camera0", "camera1", "camera2"]:
        updated_cameras[cam_id] = cameras[cam_id]
        
        if cam_id != "camera0":  # Camera0 es la referencia
            param_size = camera_param_sizes[cam_id]
            cam_params = params[param_offset:param_offset + param_size]
            param_offset += param_size
            
            # Actualizar parámetros extrínsecos
            if param_size == 6:  # rvec (3) + tvec (3)
                rvec = cam_params[:3]
                tvec = cam_params[3:6]
                
                # Actualizar cámara con nuevos parámetros
                updated_cameras[cam_id] = type(cameras[cam_id])(
                    camera_id=cameras[cam_id].camera_id,
                    K=cameras[cam_id].K.copy(),
                    dist_coeffs=cameras[cam_id].dist_coeffs.copy(),
                    R=rodrigues_to_rotation_matrix(rvec),
                    t=tvec.reshape(3, 1)
                )
    
    # Calcular residuos de reproyección de forma consistente
    for i, point_idx in enumerate(point_indices):
        point_3d = points_3d[i]
        
        for cam_id in ["camera0", "camera1", "camera2"]:
            coords, confs = frame_keypoints[cam_id]
            
            if confs[point_idx] > 0.5:  # Punto válido
                observed = coords[point_idx]
                projected = project_point(point_3d, updated_cameras[cam_id])
                
                # Siempre agregar un residuo, incluso si la proyección falla
                if np.any(np.isinf(projected)) or np.any(np.isnan(projected)):
                    residuals.extend([1000.0, 1000.0])  # Penalización alta
                else:
                    residual = observed - projected
                    residuals.extend(residual)
            else:
                # Punto no válido - agregar residuo cero para mantener consistencia
                residuals.extend([0.0, 0.0])
    
    return np.array(residuals)

def full_bundle_adjustment(points_3d_init: np.ndarray, cameras: Dict,
                          frame_keypoints: Dict, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    Bundle Adjustment completo que optimiza puntos 3D y parámetros extrínsecos.
    
    Args:
        points_3d_init: Puntos 3D iniciales (N, 3)
        cameras: Diccionario de cámaras
        frame_keypoints: Datos de keypoints por cámara
        confidence_threshold: Umbral de confianza
    
    Returns:
        Tuple[puntos_3d_optimizados, cámaras_optimizadas]
    """
    
    # Identificar puntos válidos
    valid_points = []
    coords_0, confs_0 = frame_keypoints["camera0"]
    coords_1, confs_1 = frame_keypoints["camera1"] 
    coords_2, confs_2 = frame_keypoints["camera2"]
    
    for i in range(len(points_3d_init)):
        if (not np.isnan(points_3d_init[i, 0]) and 
            confs_0[i] > confidence_threshold and
            confs_1[i] > confidence_threshold and 
            confs_2[i] > confidence_threshold):
            valid_points.append(i)
    
    if len(valid_points) == 0:
        logger.warning("No hay puntos válidos para Bundle Adjustment completo")
        return points_3d_init, cameras
    
    logger.info(f"Optimizando {len(valid_points)} puntos con Bundle Adjustment completo")
    
    # Preparar puntos 3D iniciales válidos
    valid_points_3d = points_3d_init[valid_points]
    
    # Preparar parámetros de cámaras (solo extrínsecos de camera1 y camera2)
    camera_params = []
    camera_param_sizes = {"camera0": 0}  # Camera0 es referencia
    
    for cam_id in ["camera1", "camera2"]:
        cam = cameras[cam_id]
        if hasattr(cam, 'R') and hasattr(cam, 't'):
            rvec = rotation_matrix_to_rodrigues(cam.R)
            tvec = cam.t.flatten()
            camera_params.extend(rvec)
            camera_params.extend(tvec)
            camera_param_sizes[cam_id] = 6
        else:
            camera_param_sizes[cam_id] = 0
    
    # Concatenar todos los parámetros
    x0 = np.concatenate([
        valid_points_3d.flatten(),  # Puntos 3D
        np.array(camera_params)     # Parámetros de cámaras
    ])
    
    logger.info(f"Parámetros totales a optimizar: {len(x0)} ({len(valid_points_3d)*3} puntos 3D + {len(camera_params)} parámetros cámara)")
    
    # Verificar parámetros iniciales
    if np.any(~np.isfinite(x0)):
        logger.error("Parámetros iniciales contienen NaN/Inf")
        return points_3d_init, cameras
    
    try:
        # Ejecutar optimización
        result = least_squares(
            bundle_adjustment_residual,
            x0,
            args=(cameras, frame_keypoints, np.array(valid_points), camera_param_sizes),
            method='lm',  # Levenberg-Marquardt
            max_nfev=1000,
            ftol=1e-8,
            xtol=1e-8
        )
        
        if result.success:
            logger.info(f"Bundle Adjustment completo exitoso. Costo: {result.cost:.6f}")
            
            # Extraer puntos 3D optimizados
            n_points = len(valid_points)
            optimized_points_3d = result.x[:n_points * 3].reshape(n_points, 3)
            param_offset = n_points * 3
            
            # Extraer parámetros de cámaras optimizados
            optimized_cameras = {}
            for cam_id in ["camera0", "camera1", "camera2"]:
                if cam_id == "camera0":
                    optimized_cameras[cam_id] = cameras[cam_id]
                else:
                    param_size = camera_param_sizes[cam_id]
                    if param_size > 0:
                        cam_params = result.x[param_offset:param_offset + param_size]
                        param_offset += param_size
                        
                        rvec = cam_params[:3]
                        tvec = cam_params[3:6]
                        
                        # Crear cámara optimizada
                        optimized_cameras[cam_id] = type(cameras[cam_id])(
                            camera_id=cameras[cam_id].camera_id,
                            K=cameras[cam_id].K.copy(),
                            dist_coeffs=cameras[cam_id].dist_coeffs.copy(),
                            R=rodrigues_to_rotation_matrix(rvec),
                            t=tvec.reshape(3, 1)
                        )
                    else:
                        optimized_cameras[cam_id] = cameras[cam_id]
            
            # Reconstruir array completo de puntos 3D
            final_points_3d = points_3d_init.copy()
            for i, point_idx in enumerate(valid_points):
                final_points_3d[point_idx] = optimized_points_3d[i]
            
            # Mostrar estadísticas de mejora
            initial_residual = np.sum(bundle_adjustment_residual(x0, cameras, frame_keypoints, 
                                                                np.array(valid_points), camera_param_sizes)**2)
            final_residual = result.cost
            improvement = initial_residual - final_residual
            
            logger.info(f"Mejora en residuo: {initial_residual:.6f} -> {final_residual:.6f} ({improvement:.6f})")
            
            return final_points_3d, optimized_cameras
            
        else:
            logger.warning(f"Bundle Adjustment completo falló: {result.message}")
            return points_3d_init, cameras
            
    except Exception as e:
        logger.error(f"Error en Bundle Adjustment completo: {e}")
        return points_3d_init, cameras

def print_camera_changes(original_cameras: Dict, optimized_cameras: Dict):
    """Imprime los cambios en los parámetros de las cámaras."""
    
    print(f"\n{'='*60}")
    print("CAMBIOS EN PARÁMETROS EXTRÍNSECOS")
    print(f"{'='*60}")
    
    for cam_id in ["camera1", "camera2"]:  # Camera0 es referencia
        if cam_id in original_cameras and cam_id in optimized_cameras:
            orig_cam = original_cameras[cam_id]
            opt_cam = optimized_cameras[cam_id]
            
            print(f"\n{cam_id.upper()}:")
            
            if hasattr(orig_cam, 'R') and hasattr(opt_cam, 'R'):
                # Cambios en rotación
                orig_rvec = rotation_matrix_to_rodrigues(orig_cam.R)
                opt_rvec = rotation_matrix_to_rodrigues(opt_cam.R)
                rotation_change = np.linalg.norm(opt_rvec - orig_rvec)
                print(f"  Cambio en rotación: {rotation_change:.6f} rad ({np.degrees(rotation_change):.3f}°)")
                
                # Cambios en traslación
                orig_t = orig_cam.t.flatten()
                opt_t = opt_cam.t.flatten()
                translation_change = np.linalg.norm(opt_t - orig_t)
                print(f"  Cambio en traslación: {translation_change:.6f} m")
                
                # Baseline changes
                orig_baseline = np.linalg.norm(orig_t)
                opt_baseline = np.linalg.norm(opt_t)
                baseline_change = opt_baseline - orig_baseline
                print(f"  Baseline: {orig_baseline:.3f} -> {opt_baseline:.3f} m ({baseline_change:+.6f} m)")
            else:
                print("  No hay parámetros extrínsecos disponibles")

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Tuple
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
    """
    Proyecta un punto 3D a la imagen usando los parámetros de la cámara.
    No se reutiliza el código de reprojection.py porque el algoritmo fue encontrado con este código.
    """
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
                              camera_param_sizes: Dict, reference_camera: str = "camera0",
                              confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Función de residuos para Bundle Adjustment completo con soporte para N cámaras.
    
    Args:
        params: Vector de parámetros a optimizar [puntos_3d, parámetros_cámaras]
        cameras: Diccionario de cámaras
        frame_keypoints: Keypoints por cámara
        point_indices: Índices de puntos válidos
        camera_param_sizes: Tamaños de parámetros por cámara
        reference_camera: ID de la cámara de referencia (no se optimiza)
        confidence_threshold: Umbral de confianza para considerar puntos válidos
    """
    
    residuals = []
    param_offset = 0
    
    # Extraer puntos 3D
    n_points = len(point_indices)
    points_3d = params[:n_points * 3].reshape(n_points, 3)
    param_offset += n_points * 3
    
    # Obtener lista ordenada de IDs de cámaras
    camera_ids = sorted(cameras.keys())
    
    # Extraer parámetros de cámaras y actualizar cámaras
    updated_cameras = {}
    for cam_id in camera_ids:
        if cam_id == reference_camera:
            # Cámara de referencia no se optimiza
            updated_cameras[cam_id] = cameras[cam_id]
        else:
            param_size = camera_param_sizes[cam_id]
            if param_size > 0:
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
                else:
                    updated_cameras[cam_id] = cameras[cam_id]
            else:
                updated_cameras[cam_id] = cameras[cam_id]
    
    # Calcular residuos de reproyección para todas las cámaras
    for i, point_idx in enumerate(point_indices):
        point_3d = points_3d[i]
        
        for cam_id in camera_ids:
            coords, confs = frame_keypoints[cam_id]
            
            if confs[point_idx] > confidence_threshold:  # Punto válido
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

def bundle_adjustment(points_3d_init: np.ndarray, cameras: Dict,
                          frame_keypoints: Dict, confidence_threshold: float = 0.5,
                          reference_camera: str = "camera0") -> Tuple[np.ndarray, Dict]:
    """
    Bundle Adjustment completo que optimiza puntos 3D y parámetros extrínsecos para N cámaras.
    
    Args:
        points_3d_init: Puntos 3D iniciales (N, 3)
        cameras: Diccionario de cámaras (soporta cualquier número >= 1)
        frame_keypoints: Datos de keypoints por cámara
        confidence_threshold: Umbral de confianza
        reference_camera: ID de la cámara de referencia (no se optimiza)
    
    Returns:
        Tuple[puntos_3d_optimizados, cámaras_optimizadas]
    """
    
    # Obtener lista ordenada de IDs de cámaras
    camera_ids = sorted(cameras.keys())
    
    # Verificar que la cámara de referencia existe
    if reference_camera not in camera_ids:
        logger.error(f"Cámara de referencia {reference_camera} no encontrada en {camera_ids}")
        return points_3d_init, cameras
    
    # Identificar puntos válidos en todas las cámaras
    valid_points = []
    
    # Crear máscara de validez para todas las cámaras
    for i in range(len(points_3d_init)):
        if np.isnan(points_3d_init[i, 0]):
            continue
            
        # Verificar que el punto es válido en todas las cámaras
        valid_in_all = True
        for cam_id in camera_ids:
            coords, confs = frame_keypoints[cam_id]
            if confs[i] <= confidence_threshold:
                valid_in_all = False
                break
        
        if valid_in_all:
            valid_points.append(i)
    
    if len(valid_points) == 0:
        logger.warning("No hay puntos válidos para Bundle Adjustment completo")
        return points_3d_init, cameras
    
    logger.info(f"Optimizando {len(valid_points)} puntos con Bundle Adjustment completo")
    
    # Preparar puntos 3D iniciales válidos
    valid_points_3d = points_3d_init[valid_points]
    
    # Preparar parámetros de cámaras (extrínsecos de todas excepto la de referencia)
    camera_params = []
    camera_param_sizes = {}
    
    for cam_id in camera_ids:
        if cam_id == reference_camera:
            camera_param_sizes[cam_id] = 0  # Cámara de referencia no se optimiza
        else:
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
            args=(cameras, frame_keypoints, np.array(valid_points), camera_param_sizes, 
                  reference_camera, confidence_threshold),
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
            for cam_id in camera_ids:
                if cam_id == reference_camera:
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
                                                                np.array(valid_points), camera_param_sizes,
                                                                reference_camera, confidence_threshold)**2)
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


def print_extrinsic_matrices_bundle(cameras: Dict, title: str = "PARÁMETROS EXTRÍNSECOS BUNDLE ADJUSTMENT"):
    """
    Muestra las matrices de parámetros extrínsecos optimizados por Bundle Adjustment.
    Se emplea únicamente en complete_analysis.py para mostrar resultados.
    
    Args:
        cameras: Diccionario de cámaras optimizadas
        title: Título a mostrar en el encabezado
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    camera_ids = sorted(cameras.keys())
    
    for cam_id in camera_ids:
        cam = cameras[cam_id]
        print(f"\n{cam_id.upper()}:")
        
        if hasattr(cam, 'R') and hasattr(cam, 't'):
            print("  Matriz de Rotación Optimizada (R):")
            for i, row in enumerate(cam.R):
                print(f"    [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")
            
            print("  Vector de Traslación Optimizado (t):")
            t_flat = cam.t.flatten()
            print(f"    [{t_flat[0]:8.5f}, {t_flat[1]:8.5f}, {t_flat[2]:8.5f}] metros")
            
            # Información adicional
            baseline = np.linalg.norm(cam.t)
            print(f"  Baseline optimizado: {baseline:.5f} metros")
            
            # Ángulos de Euler aproximados
            import math
            rx = math.degrees(math.atan2(cam.R[2,1], cam.R[2,2]))
            ry = math.degrees(math.atan2(-cam.R[2,0], math.sqrt(cam.R[2,1]**2 + cam.R[2,2]**2)))
            rz = math.degrees(math.atan2(cam.R[1,0], cam.R[0,0]))
            print(f"  Ángulos optimizados: Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
            
            # Información de optimización
            if hasattr(cam, 'optimization_info'):
                print(f"  Info optimización: {cam.optimization_info}")
        else:
            print("  Sin parámetros extrínsecos (cámara de referencia)")
    
    print(f"\n{'='*70}")

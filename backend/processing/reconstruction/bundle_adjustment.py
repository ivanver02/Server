"""
Bundle Adjustment para optimización de reconstrucción 3D
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
from scipy.optimize import least_squares

from .camera import CameraSystem

logger = logging.getLogger(__name__)

class BundleAdjuster:
    """
    Bundle Adjustment para optimización simultánea de puntos 3D y parámetros de cámara
    """
    
    def __init__(self, camera_system: CameraSystem):
        self.camera_system = camera_system
        self.confidence_threshold = 0.3
        self.max_iterations = 1000
        self.tolerance = 1e-8
        self.huber_threshold = 1.0  # Para pérdida robusta
    
    def _project_point(self, point_3d: np.ndarray, camera_params: np.ndarray) -> np.ndarray:
        """
        Proyectar punto 3D usando parámetros de cámara
        camera_params: [fx, fy, cx, cy, k1, k2, r1, r2, r3, t1, t2, t3]
        """
        fx, fy, cx, cy = camera_params[:4]
        k1, k2 = camera_params[4:6]
        rvec = camera_params[6:9]
        tvec = camera_params[9:12]
        
        # Convertir rodrigues a matriz de rotación
        R, _ = cv2.Rodrigues(rvec)
        
        # Transformar punto al sistema de coordenadas de la cámara
        point_cam = R @ point_3d + tvec
        
        if point_cam[2] <= 0:
            return np.array([np.inf, np.inf])
        
        # Proyección perspectiva
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]
        
        # Distorsión radial
        r2 = x*x + y*y
        distortion = 1 + k1*r2 + k2*r2*r2
        
        # Proyección final
        u = fx * x * distortion + cx
        v = fy * y * distortion + cy
        
        return np.array([u, v])
    
    def _residual_function(self, params: np.ndarray, observations: List[Tuple], 
                          camera_indices: List[int], point_indices: List[int],
                          fixed_cameras: List[int]) -> np.ndarray:
        """
        Función de residuo para optimización
        """
        # Extraer parámetros
        num_cameras = len(self.camera_system.cameras)
        num_points = len(set(point_indices))
        
        # Parámetros de cámaras (12 por cámara: intrínsecos + extrínsecos)
        camera_params_per_camera = 12
        camera_params = params[:num_cameras * camera_params_per_camera].reshape(num_cameras, -1)
        
        # Parámetros de puntos 3D
        points_3d = params[num_cameras * camera_params_per_camera:].reshape(num_points, 3)
        
        residuals = []
        
        for i, (camera_idx, point_idx, observed_2d, weight) in enumerate(observations):
            if camera_idx in fixed_cameras:
                # Usar parámetros fijos para cámaras de referencia
                camera = self.camera_system.get_camera(camera_idx)
                projected_2d = camera.project_3d_to_2d(points_3d[point_idx].reshape(1, 3))[0]
            else:
                # Usar parámetros optimizables
                projected_2d = self._project_point(points_3d[point_idx], camera_params[camera_idx])
            
            # Residuo ponderado
            residual = weight * (observed_2d - projected_2d)
            residuals.extend(residual)
        
        return np.array(residuals)
    
    def _huber_loss(self, residuals: np.ndarray) -> np.ndarray:
        """Aplicar pérdida robusta de Huber"""
        abs_residuals = np.abs(residuals)
        linear = abs_residuals <= self.huber_threshold
        
        loss = np.where(linear, 
                       0.5 * residuals**2,
                       self.huber_threshold * abs_residuals - 0.5 * self.huber_threshold**2)
        return loss
    
    def optimize_reconstruction(self, keypoints_2d: Dict[int, Dict[str, np.ndarray]], 
                              confidences_2d: Dict[int, Dict[str, np.ndarray]],
                              initial_points_3d: Dict[str, np.ndarray],
                              optimize_cameras: bool = False) -> Dict[str, np.ndarray]:
        """
        Optimizar reconstrucción 3D usando Bundle Adjustment
        
        Args:
            keypoints_2d: Dict {camera_id: {frame_key: keypoints_array}}
            confidences_2d: Dict {camera_id: {frame_key: confidence_array}}
            initial_points_3d: Dict {frame_key: points_3d_array} estimación inicial
            optimize_cameras: Si optimizar también parámetros de cámara
        
        Returns:
            optimized_points_3d: Dict {frame_key: optimized_points_3d_array}
        """
        logger.info("Iniciando Bundle Adjustment")
        
        optimized_results = {}
        
        for frame_key in initial_points_3d.keys():
            logger.debug(f"Optimizando frame {frame_key}")
            
            # Recopilar observaciones para este frame
            observations = []
            camera_indices = []
            point_indices = []
            valid_points_mask = []
            
            initial_3d = initial_points_3d[frame_key]
            num_keypoints = initial_3d.shape[0]
            
            for kp_idx in range(num_keypoints):
                if not np.isnan(initial_3d[kp_idx, 0]):  # Punto válido
                    point_observations = []
                    
                    for camera_id in keypoints_2d.keys():
                        if (frame_key in keypoints_2d[camera_id] and 
                            kp_idx < keypoints_2d[camera_id][frame_key].shape[0] and
                            kp_idx < confidences_2d[camera_id][frame_key].shape[0]):
                            
                            confidence = confidences_2d[camera_id][frame_key][kp_idx]
                            
                            if confidence > self.confidence_threshold:
                                observed_2d = keypoints_2d[camera_id][frame_key][kp_idx]
                                weight = confidence  # Ponderación por confianza
                                
                                observations.append((camera_id, len(valid_points_mask), observed_2d, weight))
                                camera_indices.append(camera_id)
                                point_indices.append(len(valid_points_mask))
                    
                    if len(point_observations) >= 2:  # Mínimo 2 observaciones
                        valid_points_mask.append(kp_idx)
                        observations.extend(point_observations)
            
            if not observations:
                logger.warning(f"No hay observaciones válidas para frame {frame_key}")
                optimized_results[frame_key] = initial_3d.copy()
                continue
            
            # Extraer puntos 3D válidos para optimización
            valid_3d_points = initial_3d[valid_points_mask]
            
            # Preparar parámetros iniciales
            if optimize_cameras:
                # Incluir parámetros de cámara en optimización
                initial_params = self._pack_camera_and_point_params(valid_3d_points)
                fixed_cameras = [0]  # Fijar cámara de referencia
            else:
                # Solo optimizar puntos 3D
                initial_params = valid_3d_points.flatten()
                fixed_cameras = list(self.camera_system.cameras.keys())
            
            # Función objetivo simplificada para solo puntos 3D
            def point_only_residual(point_params):
                points_3d = point_params.reshape(-1, 3)
                residuals = []
                
                obs_idx = 0
                for camera_id, point_idx, observed_2d, weight in observations:
                    camera = self.camera_system.get_camera(camera_id)
                    projected_2d = camera.project_3d_to_2d(points_3d[point_idx].reshape(1, 3))[0]
                    residual = weight * (observed_2d - projected_2d)
                    residuals.extend(residual)
                
                return np.array(residuals)
            
            # Optimización con scipy
            try:
                if optimize_cameras:
                    # Bundle adjustment completo (más complejo, no implementado completamente)
                    logger.warning("Optimización de cámaras no completamente implementada")
                    optimized_params = initial_params
                else:
                    # Solo optimizar puntos 3D
                    result = least_squares(
                        point_only_residual,
                        initial_params,
                        method='lm',
                        max_nfev=self.max_iterations,
                        xtol=self.tolerance,
                        ftol=self.tolerance
                    )
                    optimized_params = result.x
                
                # Reconstruir array completo de puntos 3D
                optimized_3d = initial_3d.copy()
                optimized_points = optimized_params.reshape(-1, 3)
                
                for i, original_idx in enumerate(valid_points_mask):
                    optimized_3d[original_idx] = optimized_points[i]
                
                optimized_results[frame_key] = optimized_3d
                
                # Calcular mejora en error
                initial_error = np.linalg.norm(point_only_residual(initial_params))
                final_error = np.linalg.norm(point_only_residual(optimized_params))
                improvement = (initial_error - final_error) / initial_error * 100
                
                logger.debug(f"Frame {frame_key}: error inicial={initial_error:.4f}, "
                           f"final={final_error:.4f}, mejora={improvement:.2f}%")
                
            except Exception as e:
                logger.warning(f"Error en optimización de frame {frame_key}: {e}")
                optimized_results[frame_key] = initial_3d.copy()
        
        logger.info(f"Bundle Adjustment completado para {len(optimized_results)} frames")
        return optimized_results
    
    def _pack_camera_and_point_params(self, points_3d: np.ndarray) -> np.ndarray:
        """Empaquetar parámetros de cámaras y puntos para optimización"""
        camera_params = []
        
        for camera_id in sorted(self.camera_system.cameras.keys()):
            camera = self.camera_system.get_camera(camera_id)
            
            # Parámetros intrínsecos: fx, fy, cx, cy, k1, k2
            fx, fy = camera.camera_matrix[0,0], camera.camera_matrix[1,1]
            cx, cy = camera.camera_matrix[0,2], camera.camera_matrix[1,2]
            k1, k2 = camera.distortion_coeffs[0], camera.distortion_coeffs[1]
            
            # Parámetros extrínsecos: rodrigues rotation + translation
            import cv2
            rvec, _ = cv2.Rodrigues(camera.rotation_matrix)
            
            cam_params = [fx, fy, cx, cy, k1, k2] + rvec.flatten().tolist() + camera.translation_vector.tolist()
            camera_params.extend(cam_params)
        
        # Combinar con puntos 3D
        all_params = camera_params + points_3d.flatten().tolist()
        return np.array(all_params)
    
    def set_parameters(self, confidence_threshold: float = 0.3,
                      max_iterations: int = 1000,
                      tolerance: float = 1e-8,
                      huber_threshold: float = 1.0):
        """Configurar parámetros de optimización"""
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.huber_threshold = huber_threshold
        
        logger.info(f"Parámetros Bundle Adjustment: conf_threshold={confidence_threshold}, "
                   f"max_iter={max_iterations}, tolerance={tolerance}")


def optimize_with_bundle_adjustment(camera_system: CameraSystem,
                                   keypoints_2d: Dict[int, Dict[str, np.ndarray]],
                                   confidences_2d: Dict[int, Dict[str, np.ndarray]],
                                   initial_points_3d: Dict[str, np.ndarray],
                                   optimize_cameras: bool = False,
                                   confidence_threshold: float = 0.3,
                                   max_iterations: int = 1000) -> Dict[str, np.ndarray]:
    """
    Función principal para Bundle Adjustment
    """
    import cv2  # Importar aquí para evitar problemas de dependencias
    
    adjuster = BundleAdjuster(camera_system)
    adjuster.set_parameters(confidence_threshold, max_iterations)
    
    return adjuster.optimize_reconstruction(
        keypoints_2d, confidences_2d, initial_points_3d, optimize_cameras
    )

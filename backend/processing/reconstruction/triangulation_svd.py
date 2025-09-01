"""
Triangulación 3D usando SVD (Singular Value Decomposition)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .camera import CameraSystem

logger = logging.getLogger(__name__)

class SVDTriangulator:
    """
    Triangulador 3D usando descomposición en valores singulares
    """
    
    def __init__(self, camera_system: CameraSystem):
        self.camera_system = camera_system
        self.min_cameras = 2
        self.max_reprojection_error = 5.0  # píxeles
    
    def triangulate_point(self, observations: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """
        Triangular un punto 3D a partir de observaciones 2D en múltiples cámaras
        
        Args:
            observations: Dict {camera_id: punto_2d} donde punto_2d es array [x, y]
        
        Returns:
            punto_3d: array [X, Y, Z] o None si falla la triangulación
        """
        if len(observations) < self.min_cameras:
            return None
        
        # Obtener cámaras calibradas
        calibrated_cameras = self.camera_system.get_calibrated_cameras()
        
        # Filtrar observaciones de cámaras calibradas
        valid_observations = {cam_id: point for cam_id, point in observations.items() 
                            if cam_id in calibrated_cameras}
        
        if len(valid_observations) < self.min_cameras:
            return None
        
        # Construir sistema de ecuaciones A * X = 0
        A = []
        
        for camera_id, point_2d in valid_observations.items():
            camera = calibrated_cameras[camera_id]
            P = camera.get_projection_matrix()
            
            x, y = point_2d[0], point_2d[1]
            
            # Ecuaciones de triangulación DLT (Direct Linear Transform)
            # x * P[2,:] - P[0,:] = 0
            # y * P[2,:] - P[1,:] = 0
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        
        # Resolver usando SVD: A * X = 0
        try:
            _, _, Vt = np.linalg.svd(A)
            X_homogeneous = Vt[-1, :]
            
            # Convertir de coordenadas homogéneas a 3D
            if abs(X_homogeneous[3]) < 1e-8:
                return None
            
            point_3d = X_homogeneous[:3] / X_homogeneous[3]
            
            # Verificar que el punto está delante de todas las cámaras
            if not self._is_point_in_front_of_cameras(point_3d, valid_observations.keys()):
                return None
            
            # Verificar error de reproyección
            reprojection_error = self._calculate_reprojection_error(point_3d, valid_observations)
            if reprojection_error > self.max_reprojection_error:
                logger.debug(f"Error de reproyección alto: {reprojection_error:.2f} px")
                return None
            
            return point_3d
            
        except np.linalg.LinAlgError:
            logger.debug("Error en SVD durante triangulación")
            return None
    
    def _is_point_in_front_of_cameras(self, point_3d: np.ndarray, camera_ids: List[int]) -> bool:
        """Verificar que el punto está delante de todas las cámaras"""
        for camera_id in camera_ids:
            camera = self.camera_system.get_camera(camera_id)
            
            # Transformar punto a sistema de coordenadas de la cámara
            if camera.is_reference:
                point_cam = point_3d
            else:
                point_cam = camera.rotation_matrix @ point_3d + camera.translation_vector
            
            # Verificar que Z > 0 (delante de la cámara)
            if point_cam[2] <= 0:
                return False
        
        return True
    
    def _calculate_reprojection_error(self, point_3d: np.ndarray, 
                                    observations: Dict[int, np.ndarray]) -> float:
        """Calcular error RMS de reproyección"""
        total_error = 0.0
        num_observations = 0
        
        for camera_id, observed_2d in observations.items():
            camera = self.camera_system.get_camera(camera_id)
            projected_2d = camera.project_3d_to_2d(point_3d.reshape(1, 3))
            
            if projected_2d.shape[0] > 0:
                error = np.linalg.norm(observed_2d - projected_2d[0])
                total_error += error * error
                num_observations += 1
        
        if num_observations == 0:
            return float('inf')
        
        return np.sqrt(total_error / num_observations)
    
    def triangulate_frame(self, frame_keypoints: Dict[int, np.ndarray], 
                         confidences: Dict[int, np.ndarray],
                         confidence_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangular todos los keypoints de un frame
        
        Args:
            frame_keypoints: Dict {camera_id: keypoints_array} con forma (N, 2)
            confidences: Dict {camera_id: confidence_array} con forma (N,)
            confidence_threshold: Umbral mínimo de confianza
        
        Returns:
            points_3d: array (M, 3) con puntos triangulados
            point_confidences: array (M,) con confianzas promedio
        """
        if not frame_keypoints:
            return np.array([]).reshape(0, 3), np.array([])
        
        # Determinar número de keypoints (asumir que todas las cámaras tienen el mismo número)
        num_keypoints = max(kp.shape[0] for kp in frame_keypoints.values())
        
        points_3d = []
        point_confidences = []
        
        for kp_idx in range(num_keypoints):
            # Recopilar observaciones de este keypoint en todas las cámaras
            observations = {}
            kp_confidences = []
            
            for camera_id, keypoints in frame_keypoints.items():
                if kp_idx < keypoints.shape[0] and kp_idx < confidences[camera_id].shape[0]:
                    confidence = confidences[camera_id][kp_idx]
                    
                    if confidence > confidence_threshold:
                        observations[camera_id] = keypoints[kp_idx]
                        kp_confidences.append(confidence)
            
            # Triangular si hay suficientes observaciones
            if len(observations) >= self.min_cameras:
                point_3d = self.triangulate_point(observations)
                
                if point_3d is not None:
                    points_3d.append(point_3d)
                    point_confidences.append(np.mean(kp_confidences))
                else:
                    # Punto no triangulable, usar NaN
                    points_3d.append(np.array([np.nan, np.nan, np.nan]))
                    point_confidences.append(0.0)
            else:
                # Insuficientes observaciones
                points_3d.append(np.array([np.nan, np.nan, np.nan]))
                point_confidences.append(0.0)
        
        return np.array(points_3d), np.array(point_confidences)
    
    def triangulate_sequence(self, keypoints_2d: Dict[int, Dict[str, np.ndarray]], 
                           confidences_2d: Dict[int, Dict[str, np.ndarray]],
                           confidence_threshold: float = 0.3) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Triangular secuencia completa de frames
        
        Args:
            keypoints_2d: Dict {camera_id: {frame_key: keypoints_array}}
            confidences_2d: Dict {camera_id: {frame_key: confidence_array}}
            confidence_threshold: Umbral mínimo de confianza
        
        Returns:
            results: Dict {frame_key: (points_3d, confidences)}
        """
        # Obtener lista de frames comunes
        all_frames = set()
        for camera_data in keypoints_2d.values():
            all_frames.update(camera_data.keys())
        
        results = {}
        
        for frame_key in sorted(all_frames):
            # Recopilar keypoints de este frame de todas las cámaras
            frame_keypoints = {}
            frame_confidences = {}
            
            for camera_id in keypoints_2d.keys():
                if frame_key in keypoints_2d[camera_id]:
                    frame_keypoints[camera_id] = keypoints_2d[camera_id][frame_key]
                    frame_confidences[camera_id] = confidences_2d[camera_id][frame_key]
            
            if frame_keypoints:
                points_3d, point_confidences = self.triangulate_frame(
                    frame_keypoints, frame_confidences, confidence_threshold
                )
                results[frame_key] = (points_3d, point_confidences)
                
                # Log estadísticas
                valid_points = np.sum(~np.isnan(points_3d[:, 0]))
                logger.debug(f"Frame {frame_key}: {valid_points}/{len(points_3d)} puntos triangulados")
        
        logger.info(f"Triangulación SVD completada para {len(results)} frames")
        return results
    
    def set_parameters(self, min_cameras: int = 2, max_reprojection_error: float = 5.0):
        """Configurar parámetros de triangulación"""
        self.min_cameras = min_cameras
        self.max_reprojection_error = max_reprojection_error
        logger.info(f"Parámetros SVD: min_cameras={min_cameras}, max_error={max_reprojection_error}")


def triangulate_with_svd(camera_system: CameraSystem,
                        keypoints_2d: Dict[int, Dict[str, np.ndarray]],
                        confidences_2d: Dict[int, Dict[str, np.ndarray]],
                        confidence_threshold: float = 0.3,
                        min_cameras: int = 2,
                        max_reprojection_error: float = 5.0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Función principal para triangulación con SVD
    """
    triangulator = SVDTriangulator(camera_system)
    triangulator.set_parameters(min_cameras, max_reprojection_error)
    
    return triangulator.triangulate_sequence(keypoints_2d, confidences_2d, confidence_threshold)

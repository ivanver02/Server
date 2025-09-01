"""
Cálculo de parámetros extrínsecos usando keypoints 2D correspondientes
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize, least_squares
import cv2

from .camera import CameraSystem

logger = logging.getLogger(__name__)

class ExtrinsicsCalculator:
    """
    Calculador de parámetros extrínsecos usando correspondencias de keypoints 2D
    """
    
    def __init__(self, camera_system: CameraSystem):
        self.camera_system = camera_system
        self.reference_camera_id = 0
        
    def find_corresponding_keypoints(self, keypoints_2d: Dict[int, Dict[str, np.ndarray]], 
                                   confidences_2d: Dict[int, Dict[str, np.ndarray]],
                                   min_confidence: float = 0.3) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Encontrar keypoints correspondientes entre cámaras en los mismos frames
        
        Args:
            keypoints_2d: {camera_id: {frame_key: coordinates}}
            confidences_2d: {camera_id: {frame_key: confidences}}
            min_confidence: Confianza mínima para considerar un keypoint válido
            
        Returns:
            Dict {frame_key: {camera_id: valid_keypoints}}
        """
        corresponding_keypoints = {}
        
        # Obtener frames comunes a todas las cámaras
        all_camera_ids = list(keypoints_2d.keys())
        if not all_camera_ids:
            return corresponding_keypoints
            
        # Frames disponibles en la primera cámara
        common_frames = set(keypoints_2d[all_camera_ids[0]].keys())
        
        # Intersección con frames de otras cámaras
        for camera_id in all_camera_ids[1:]:
            camera_frames = set(keypoints_2d[camera_id].keys())
            common_frames = common_frames.intersection(camera_frames)
        
        logger.info(f"Encontrados {len(common_frames)} frames comunes entre {len(all_camera_ids)} cámaras")
        
        # Para cada frame común, extraer keypoints válidos
        for frame_key in common_frames:
            frame_keypoints = {}
            
            for camera_id in all_camera_ids:
                if frame_key in keypoints_2d[camera_id] and frame_key in confidences_2d[camera_id]:
                    coordinates = keypoints_2d[camera_id][frame_key]
                    confidences = confidences_2d[camera_id][frame_key]
                    
                    # Filtrar por confianza
                    valid_mask = confidences >= min_confidence
                    if np.any(valid_mask):
                        valid_coordinates = coordinates[valid_mask]
                        frame_keypoints[camera_id] = valid_coordinates
            
            # Solo incluir frames donde todas las cámaras tienen keypoints válidos
            if len(frame_keypoints) == len(all_camera_ids):
                corresponding_keypoints[frame_key] = frame_keypoints
        
        logger.info(f"Keypoints correspondientes válidos en {len(corresponding_keypoints)} frames")
        return corresponding_keypoints
    
    def calculate_extrinsics_pnp(self, camera_id: int, 
                                correspondences: Dict[str, Dict[int, np.ndarray]]) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Calcular extrínsecos usando PnP (Perspective-n-Point) con triangulación previa
        
        Args:
            camera_id: ID de la cámara a calibrar
            correspondences: Correspondencias de keypoints válidas
            
        Returns:
            (success, rotation_matrix, translation_vector)
        """
        if camera_id == self.reference_camera_id:
            # La cámara de referencia tiene extrínsecos identidad
            return True, np.eye(3), np.zeros(3)
        
        reference_camera = self.camera_system.get_camera(self.reference_camera_id)
        target_camera = self.camera_system.get_camera(camera_id)
        
        if not reference_camera or not target_camera:
            logger.error(f"Cámara {self.reference_camera_id} o {camera_id} no encontrada")
            return False, np.eye(3), np.zeros(3)
        
        # Recolectar puntos 2D y 3D correspondientes
        points_2d_ref = []
        points_2d_target = []
        
        # Usar múltiples frames para mayor robustez
        for frame_key, frame_keypoints in correspondences.items():
            if self.reference_camera_id in frame_keypoints and camera_id in frame_keypoints:
                ref_points = frame_keypoints[self.reference_camera_id]
                target_points = frame_keypoints[camera_id]
                
                # Asegurar mismo número de puntos
                min_points = min(len(ref_points), len(target_points))
                ref_points = ref_points[:min_points]
                target_points = target_points[:min_points]
                
                points_2d_ref.extend(ref_points)
                points_2d_target.extend(target_points)
        
        if len(points_2d_ref) < 10:
            logger.error(f"Insuficientes correspondencias para cámara {camera_id}: {len(points_2d_ref)}")
            return False, np.eye(3), np.zeros(3)
        
        points_2d_ref = np.array(points_2d_ref)
        points_2d_target = np.array(points_2d_target)
        
        # Paso 1: Triangular puntos 3D usando cámara de referencia
        # Asumir que la cámara de referencia ve los puntos a cierta distancia promedio
        points_3d = self._estimate_3d_points_from_reference(points_2d_ref, reference_camera)
        
        # Paso 2: Usar PnP para encontrar pose de la cámara target
        success, rvec, tvec = cv2.solvePnP(
            points_3d.astype(np.float32),
            points_2d_target.astype(np.float32),
            target_camera.camera_matrix,
            target_camera.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convertir vector de rotación a matriz
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            translation_vector = tvec.flatten()
            
            logger.info(f"Extrínsecos calculados para cámara {camera_id} usando PnP")
            return True, rotation_matrix, translation_vector
        else:
            logger.error(f"PnP falló para cámara {camera_id}")
            return False, np.eye(3), np.zeros(3)
    
    def _estimate_3d_points_from_reference(self, points_2d: np.ndarray, 
                                         reference_camera) -> np.ndarray:
        """
        Estimar puntos 3D asumiendo que están en un plano a distancia promedio
        """
        # Distancia promedio estimada (metros) - ajustable según el setup
        average_depth = 2.0  # 2 metros
        
        # Deshacer proyección asumiendo profundidad constante
        points_3d = []
        
        for point_2d in points_2d:
            # Convertir a coordenadas normalizadas
            x_norm = (point_2d[0] - reference_camera.camera_matrix[0, 2]) / reference_camera.camera_matrix[0, 0]
            y_norm = (point_2d[1] - reference_camera.camera_matrix[1, 2]) / reference_camera.camera_matrix[1, 1]
            
            # Proyectar a 3D con profundidad constante
            point_3d = np.array([
                x_norm * average_depth,
                y_norm * average_depth,
                average_depth
            ])
            points_3d.append(point_3d)
        
        return np.array(points_3d)
    
    def calculate_extrinsics_optimization(self, camera_id: int,
                                        correspondences: Dict[str, Dict[int, np.ndarray]]) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Calcular extrínsecos usando optimización no lineal
        """
        if camera_id == self.reference_camera_id:
            return True, np.eye(3), np.zeros(3)
        
        # Recolectar correspondencias
        ref_points_all = []
        target_points_all = []
        
        for frame_key, frame_keypoints in correspondences.items():
            if self.reference_camera_id in frame_keypoints and camera_id in frame_keypoints:
                ref_points = frame_keypoints[self.reference_camera_id]
                target_points = frame_keypoints[camera_id]
                
                min_points = min(len(ref_points), len(target_points))
                ref_points_all.extend(ref_points[:min_points])
                target_points_all.extend(target_points[:min_points])
        
        if len(ref_points_all) < 8:
            logger.error(f"Insuficientes correspondencias: {len(ref_points_all)}")
            return False, np.eye(3), np.zeros(3)
        
        ref_points = np.array(ref_points_all)
        target_points = np.array(target_points_all)
        
        # Obtener cámaras
        ref_camera = self.camera_system.get_camera(self.reference_camera_id)
        target_camera = self.camera_system.get_camera(camera_id)
        
        # Parámetros iniciales [rx, ry, rz, tx, ty, tz]
        initial_params = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        
        def residual_function(params):
            """Función de residuos para optimización"""
            rx, ry, rz, tx, ty, tz = params
            
            # Construir matriz de rotación y vector de traslación
            rvec = np.array([rx, ry, rz])
            tvec = np.array([tx, ty, tz])
            R, _ = cv2.Rodrigues(rvec)
            
            residuals = []
            
            # Para cada correspondencia, calcular error de reproyección
            for i in range(len(ref_points)):
                # Estimar punto 3D desde cámara de referencia
                point_3d = self._estimate_3d_points_from_reference(
                    ref_points[i:i+1], ref_camera
                )[0]
                
                # Proyectar a cámara target
                projected, _ = cv2.projectPoints(
                    point_3d.reshape(1, 1, 3),
                    rvec, tvec,
                    target_camera.camera_matrix,
                    target_camera.distortion_coeffs
                )
                projected_2d = projected.reshape(2)
                
                # Error de reproyección
                error = target_points[i] - projected_2d
                residuals.extend(error)
            
            return np.array(residuals)
        
        # Optimización
        try:
            result = least_squares(residual_function, initial_params, 
                                 method='lm', max_nfev=1000)
            
            if result.success:
                rx, ry, rz, tx, ty, tz = result.x
                rvec = np.array([rx, ry, rz])
                tvec = np.array([tx, ty, tz])
                
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                translation_vector = tvec
                
                logger.info(f"Extrínsecos optimizados para cámara {camera_id}")
                return True, rotation_matrix, translation_vector
            else:
                logger.error(f"Optimización falló para cámara {camera_id}")
                return False, np.eye(3), np.zeros(3)
                
        except Exception as e:
            logger.error(f"Error en optimización para cámara {camera_id}: {e}")
            return False, np.eye(3), np.zeros(3)
    
    def calibrate_camera_system(self, keypoints_2d: Dict[int, Dict[str, np.ndarray]], 
                               confidences_2d: Dict[int, Dict[str, np.ndarray]],
                               method: str = "pnp",
                               min_confidence: float = 0.3) -> bool:
        """
        Calibrar sistema completo de cámaras
        
        Args:
            keypoints_2d: Keypoints 2D por cámara y frame
            confidences_2d: Confianzas por cámara y frame  
            method: Método de cálculo ("pnp" o "optimization")
            min_confidence: Confianza mínima para keypoints válidos
        """
        logger.info(f"Iniciando calibración de extrínsecos con método: {method}")
        
        # Encontrar correspondencias válidas
        correspondences = self.find_corresponding_keypoints(
            keypoints_2d, confidences_2d, min_confidence
        )
        
        if not correspondences:
            logger.error("No se encontraron correspondencias válidas")
            return False
        
        # Establecer cámara de referencia
        camera_ids = list(keypoints_2d.keys())
        if self.reference_camera_id not in camera_ids:
            self.reference_camera_id = camera_ids[0]
        
        self.camera_system.set_reference_camera(self.reference_camera_id)
        
        # Calibrar cada cámara
        success_count = 0
        for camera_id in camera_ids:
            if camera_id == self.reference_camera_id:
                success_count += 1
                continue
            
            if method == "pnp":
                success, R, t = self.calculate_extrinsics_pnp(camera_id, correspondences)
            elif method == "optimization":
                success, R, t = self.calculate_extrinsics_optimization(camera_id, correspondences)
            else:
                logger.error(f"Método no válido: {method}")
                return False
            
            if success:
                camera = self.camera_system.get_camera(camera_id)
                if camera:
                    camera.set_extrinsics(R, t)
                    success_count += 1
                    logger.info(f"Cámara {camera_id} calibrada: t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            else:
                logger.error(f"Falló calibración de cámara {camera_id}")
        
        total_cameras = len(camera_ids)
        logger.info(f"Calibración completada: {success_count}/{total_cameras} cámaras")
        
        return success_count == total_cameras


def calibrate_extrinsics_from_keypoints(camera_system: CameraSystem,
                                       keypoints_2d: Dict[int, Dict[str, np.ndarray]], 
                                       confidences_2d: Dict[int, Dict[str, np.ndarray]],
                                       method: str = "pnp",
                                       min_confidence: float = 0.3) -> bool:
    """
    Función principal para calibrar extrínsecos desde keypoints 2D
    """
    calculator = ExtrinsicsCalculator(camera_system)
    
    return calculator.calibrate_camera_system(
        keypoints_2d, confidences_2d, method, min_confidence
    )

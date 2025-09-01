"""
Calibración de parámetros intrínsecos y extrínsecos de cámaras
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from .camera import Camera, CameraSystem
from config.camera_intrinsics import CAMERA_INTRINSICS

logger = logging.getLogger(__name__)

class CameraCalibrator:
    """
    Calibrador de sistema multi-cámara usando patrones de calibración
    """
    
    def __init__(self, base_data_dir: Path):
        self.base_data_dir = Path(base_data_dir)
        self.camera_system = CameraSystem()
        
        # Parámetros del patrón de calibración (tablero de ajedrez)
        self.chessboard_size = (9, 6)  # Esquinas internas
        self.square_size = 25.0  # Tamaño del cuadrado en mm
        
        # Generar puntos 3D del patrón
        self.object_points_pattern = self._generate_object_points()
        
    def _generate_object_points(self) -> np.ndarray:
        """Generar puntos 3D del patrón de calibración"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def initialize_cameras_from_config(self, camera_ids: List[int], reference_id: int = 0):
        """Inicializar cámaras usando configuración predeterminada"""
        # Usar el nuevo método del CameraSystem
        self.camera_system.initialize_from_config(camera_ids, reference_id)
        logger.info(f"Sistema de cámaras inicializado desde configuración para IDs: {camera_ids}")
    
    def get_camera_system(self) -> CameraSystem:
        """Obtener el sistema de cámaras"""
        return self.camera_system
    
    def detect_chessboard_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detectar esquinas del tablero de ajedrez en una imagen"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detectar esquinas
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refinar coordenadas de esquinas
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
        return ret, corners
    
    def calibrate_single_camera(self, camera_id: int, images: List[np.ndarray]) -> bool:
        """
        Calibrar una sola cámara usando imágenes de calibración
        """
        if camera_id not in self.camera_system.cameras:
            logger.error(f"Cámara {camera_id} no encontrada en el sistema")
            return False
        
        camera = self.camera_system.cameras[camera_id]
        
        # Detectar esquinas en todas las imágenes
        object_points = []
        image_points = []
        
        valid_images = 0
        for i, image in enumerate(images):
            ret, corners = self.detect_chessboard_corners(image)
            
            if ret:
                object_points.append(self.object_points_pattern)
                image_points.append(corners.reshape(-1, 2))
                valid_images += 1
                logger.debug(f"Esquinas detectadas en imagen {i} de cámara {camera_id}")
            else:
                logger.warning(f"No se detectaron esquinas en imagen {i} de cámara {camera_id}")
        
        if valid_images < 10:
            logger.error(f"Insuficientes imágenes válidas para cámara {camera_id}: {valid_images}/10")
            return False
        
        # Calibración con OpenCV
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, camera.resolution, None, None,
            flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
        )
        
        if ret:
            # Actualizar parámetros de la cámara
            camera.camera_matrix = camera_matrix
            camera.distortion_coeffs = dist_coeffs
            
            # Calcular error de reproyección
            total_error = 0
            for i in range(len(object_points)):
                projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], 
                                               camera_matrix, dist_coeffs)
                error = cv2.norm(image_points[i], projected.reshape(-1, 2), cv2.NORM_L2) / len(projected)
                total_error += error
            
            mean_error = total_error / len(object_points)
            logger.info(f"Cámara {camera_id} calibrada. Error promedio: {mean_error:.4f} píxeles")
            
            return True
        else:
            logger.error(f"Falló la calibración de cámara {camera_id}")
            return False
    
    def get_camera_system(self) -> CameraSystem:
        """Obtener el sistema de cámaras"""
        return self.camera_system

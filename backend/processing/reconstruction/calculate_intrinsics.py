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
    
    def initialize_cameras_from_config(self, camera_ids: List[int]):
        """Inicializar cámaras usando configuración predeterminada"""
        for camera_id in camera_ids:
            camera_key = f"camera_{camera_id}"
            
            if camera_key in CAMERA_INTRINSICS:
                config = CAMERA_INTRINSICS[camera_key]
                camera = Camera(
                    camera_id=camera_id,
                    camera_matrix=config["camera_matrix"],
                    distortion_coeffs=config["distortion_coeffs"],
                    serial_number=config["serial_number"],
                    resolution=config["resolution"],
                    model=config["model"]
                )
                self.camera_system.add_camera(camera)
            else:
                logger.warning(f"No se encontró configuración para camera_{camera_id}")
    
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
    
    def calibrate_stereo_extrinsics(self, camera_id_1: int, camera_id_2: int, 
                                   image_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """
        Calibrar parámetros extrínsecos entre dos cámaras
        """
        if camera_id_1 not in self.camera_system.cameras or camera_id_2 not in self.camera_system.cameras:
            logger.error("Una o ambas cámaras no están en el sistema")
            return False
        
        camera1 = self.camera_system.cameras[camera_id_1]
        camera2 = self.camera_system.cameras[camera_id_2]
        
        # Detectar esquinas en pares de imágenes
        object_points = []
        image_points_1 = []
        image_points_2 = []
        
        valid_pairs = 0
        for i, (img1, img2) in enumerate(image_pairs):
            ret1, corners1 = self.detect_chessboard_corners(img1)
            ret2, corners2 = self.detect_chessboard_corners(img2)
            
            if ret1 and ret2:
                object_points.append(self.object_points_pattern)
                image_points_1.append(corners1.reshape(-1, 2))
                image_points_2.append(corners2.reshape(-1, 2))
                valid_pairs += 1
        
        if valid_pairs < 10:
            logger.error(f"Insuficientes pares válidos: {valid_pairs}/10")
            return False
        
        # Calibración estéreo
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            object_points, image_points_1, image_points_2,
            camera1.camera_matrix, camera1.distortion_coeffs,
            camera2.camera_matrix, camera2.distortion_coeffs,
            camera1.resolution, criteria=criteria, flags=flags
        )
        
        if ret:
            # Si camera_id_1 es la referencia, camera_id_2 obtiene R,T
            # Si camera_id_2 es la referencia, camera_id_1 obtiene R^T, -R^T*T
            if camera1.is_reference:
                camera2.set_extrinsics(R, T.flatten())
            elif camera2.is_reference:
                camera1.set_extrinsics(R.T, -R.T @ T.flatten())
            else:
                logger.warning("Ninguna cámara es de referencia, usando camera_id_1 como referencia")
                camera1.is_reference = True
                camera1.is_calibrated = True
                camera2.set_extrinsics(R, T.flatten())
            
            logger.info(f"Calibración estéreo exitosa entre cámaras {camera_id_1} y {camera_id_2}")
            return True
        else:
            logger.error(f"Falló la calibración estéreo entre cámaras {camera_id_1} y {camera_id_2}")
            return False
    
    def calibrate_multi_camera_system(self, calibration_images: Dict[int, List[np.ndarray]]) -> bool:
        """
        Calibrar sistema completo de múltiples cámaras
        """
        # 1. Calibrar cada cámara individualmente
        for camera_id, images in calibration_images.items():
            if not self.calibrate_single_camera(camera_id, images):
                logger.error(f"Falló la calibración individual de cámara {camera_id}")
                return False
        
        # 2. Calibrar parámetros extrínsecos por pares
        camera_ids = list(calibration_images.keys())
        reference_id = 0 if 0 in camera_ids else camera_ids[0]
        
        # Crear pares de imágenes para calibración estéreo
        for camera_id in camera_ids:
            if camera_id != reference_id:
                # Crear pares de imágenes sincronizadas
                min_images = min(len(calibration_images[reference_id]), 
                               len(calibration_images[camera_id]))
                
                image_pairs = [(calibration_images[reference_id][i], 
                              calibration_images[camera_id][i]) 
                              for i in range(min_images)]
                
                if not self.calibrate_stereo_extrinsics(reference_id, camera_id, image_pairs):
                    logger.error(f"Falló la calibración estéreo entre {reference_id} y {camera_id}")
                    return False
        
        logger.info("Sistema multi-cámara calibrado exitosamente")
        return True
    
    def load_calibration_images_from_directory(self, calibration_dir: Path) -> Dict[int, List[np.ndarray]]:
        """
        Cargar imágenes de calibración desde directorio
        Estructura esperada: calibration_dir/camera_X/image_*.jpg
        """
        calibration_images = {}
        
        for camera_dir in calibration_dir.iterdir():
            if camera_dir.is_dir() and camera_dir.name.startswith('camera_'):
                try:
                    camera_id = int(camera_dir.name.split('_')[1])
                    images = []
                    
                    for img_file in sorted(camera_dir.glob('*.jpg')) + sorted(camera_dir.glob('*.png')):
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            images.append(img)
                    
                    if images:
                        calibration_images[camera_id] = images
                        logger.info(f"Cargadas {len(images)} imágenes para cámara {camera_id}")
                    
                except ValueError:
                    logger.warning(f"Nombre de directorio inválido: {camera_dir.name}")
        
        return calibration_images
    
    def save_calibration(self, save_dir: Path):
        """Guardar calibración del sistema"""
        self.camera_system.save_system(save_dir)
    
    def load_calibration(self, load_dir: Path):
        """Cargar calibración del sistema"""
        self.camera_system = CameraSystem.load_system(load_dir)
    
    def get_camera_system(self) -> CameraSystem:
        """Obtener sistema de cámaras calibrado"""
        return self.camera_system


def calibrate_from_images(base_data_dir: Path, calibration_dir: Path, 
                         camera_ids: List[int]) -> CameraSystem:
    """
    Función principal para calibrar sistema desde imágenes
    """
    calibrator = CameraCalibrator(base_data_dir)
    
    # Inicializar cámaras
    calibrator.initialize_cameras_from_config(camera_ids)
    
    # Cargar imágenes de calibración
    calibration_images = calibrator.load_calibration_images_from_directory(calibration_dir)
    
    if not calibration_images:
        logger.error("No se encontraron imágenes de calibración")
        return calibrator.get_camera_system()
    
    # Calibrar sistema
    if calibrator.calibrate_multi_camera_system(calibration_images):
        # Guardar calibración
        save_dir = base_data_dir / "calibration"
        calibrator.save_calibration(save_dir)
        logger.info(f"Calibración guardada en {save_dir}")
    
    return calibrator.get_camera_system()

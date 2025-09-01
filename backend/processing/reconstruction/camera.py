"""
Clase Camera para manejo de parámetros intrínsecos y extrínsecos
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class Camera:
    """
    Clase que representa una cámara con sus parámetros intrínsecos y extrínsecos
    """
    
    def __init__(self, camera_id: int, camera_matrix: Optional[np.ndarray] = None, 
                 distortion_coeffs: Optional[np.ndarray] = None, 
                 serial_number: Optional[str] = None, resolution: Optional[Tuple[int, int]] = None, 
                 model: Optional[str] = None):
        
        self.camera_id = camera_id
        
        # Si no se proporcionan parámetros, cargar desde configuración
        if camera_matrix is None or distortion_coeffs is None:
            intrinsics = self._load_intrinsics_from_config(camera_id)
            self.camera_matrix = intrinsics['camera_matrix'].copy()
            self.distortion_coeffs = intrinsics['distortion_coeffs'].copy()
            self.serial_number = intrinsics['serial_number']
            self.resolution = intrinsics['resolution']
            self.model = intrinsics['model']
        else:
            self.camera_matrix = camera_matrix.copy()
            self.distortion_coeffs = distortion_coeffs.copy()
            self.serial_number = serial_number or f"Unknown_{camera_id}"
            self.resolution = resolution or (640, 480)
            self.model = model or "Unknown"
        
        # Parámetros extrínsecos (se calculan durante calibración del sistema)
        self.rotation_matrix = np.eye(3, dtype=np.float64)
        self.translation_vector = np.zeros(3, dtype=np.float64)
        self.is_calibrated = False  # Los extrínsecos NO están calibrados por defecto
        
        # La cámara de referencia se establece durante la calibración del sistema
        self.is_reference = False
        
        logger.info(f"Cámara {camera_id} inicializada con parámetros intrínsecos")
    
    def _load_intrinsics_from_config(self, camera_id: int) -> Dict:
        """Cargar parámetros intrínsecos desde configuración"""
        try:
            from config.camera_intrinsics import CAMERA_INTRINSICS
            camera_key = f"camera_{camera_id}"
            
            if camera_key in CAMERA_INTRINSICS:
                logger.info(f"Cargando parámetros intrínsecos para cámara {camera_id} desde configuración")
                return CAMERA_INTRINSICS[camera_key]
            else:
                logger.warning(f"No se encontraron parámetros para cámara {camera_id}, usando valores por defecto")
                # Valores por defecto para Orbbec Gemini 335Le
                return {
                    'camera_matrix': np.array([
                        [618.2, 0.0, 320.0],
                        [0.0, 618.2, 240.0],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64),
                    'distortion_coeffs': np.array([0.08, -0.15, 0.001, -0.002, 0.03], dtype=np.float64),
                    'serial_number': f"Unknown_{camera_id}",
                    'resolution': (640, 480),
                    'model': "Orbbec Gemini 335Le"
                }
        except ImportError as e:
            logger.error(f"Error importando configuración de cámaras: {e}")
            raise
    
    def set_as_reference(self):
        """Establecer esta cámara como referencia del sistema"""
        self.is_reference = True
        self.rotation_matrix = np.eye(3, dtype=np.float64)
        self.translation_vector = np.zeros(3, dtype=np.float64)
        self.is_calibrated = True
        logger.info(f"Cámara {self.camera_id} establecida como referencia del sistema")
    
    def set_extrinsics(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray):
        """Establecer parámetros extrínsecos respecto a cámara de referencia"""
        self.rotation_matrix = rotation_matrix.copy()
        self.translation_vector = translation_vector.copy()
        self.is_calibrated = True
        logger.info(f"Parámetros extrínsecos establecidos para cámara {self.camera_id}")
    
    def get_projection_matrix(self) -> np.ndarray:
        """Obtener matriz de proyección P = K[R|t]"""
        if not self.is_calibrated:
            logger.warning(f"Cámara {self.camera_id} no tiene extrínsecos calibrados")
        
        # Construir matriz [R|t]
        extrinsic_matrix = np.hstack((self.rotation_matrix, self.translation_vector.reshape(3, 1)))
        
        # P = K[R|t]
        projection_matrix = self.camera_matrix @ extrinsic_matrix
        
        return projection_matrix
    
    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Corregir distorsión de puntos 2D
        points_2d: array de forma (N, 2) con coordenadas (x, y)
        """
        import cv2
        
        if points_2d.shape[0] == 0:
            return points_2d
        
        # OpenCV requiere formato (N, 1, 2)
        points_reshaped = points_2d.reshape(-1, 1, 2).astype(np.float32)
        
        # Corregir distorsión
        undistorted = cv2.undistortPoints(
            points_reshaped, 
            self.camera_matrix, 
            self.distortion_coeffs,
            P=self.camera_matrix
        )
        
        # Retornar formato (N, 2)
        return undistorted.reshape(-1, 2)
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Proyectar puntos 3D a coordenadas 2D de esta cámara
        points_3d: array de forma (N, 3) con coordenadas (X, Y, Z)
        """
        import cv2
        
        if points_3d.shape[0] == 0:
            return np.array([]).reshape(0, 2)
        
        # Proyectar usando cv2.projectPoints
        points_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3).astype(np.float32),
            self.rotation_matrix,
            self.translation_vector,
            self.camera_matrix,
            self.distortion_coeffs
        )
        
        return points_2d.reshape(-1, 2)
    
    def get_camera_center(self) -> np.ndarray:
        """Obtener centro de la cámara en coordenadas del mundo"""
        if self.is_reference:
            return np.zeros(3)
        
        # C = -R^T * t
        camera_center = -self.rotation_matrix.T @ self.translation_vector
        return camera_center
    
    def save_calibration(self, save_path: Path):
        """Guardar parámetros de calibración"""
        calibration_data = {
            'camera_id': self.camera_id,
            'camera_matrix': self.camera_matrix,
            'distortion_coeffs': self.distortion_coeffs,
            'rotation_matrix': self.rotation_matrix,
            'translation_vector': self.translation_vector,
            'serial_number': self.serial_number,
            'resolution': self.resolution,
            'model': self.model,
            'is_calibrated': self.is_calibrated,
            'is_reference': self.is_reference
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **calibration_data)
        logger.info(f"Calibración de cámara {self.camera_id} guardada en {save_path}")
    
    @classmethod
    def load_calibration(cls, load_path: Path) -> 'Camera':
        """Cargar parámetros de calibración"""
        data = np.load(load_path, allow_pickle=True)
        
        camera = cls(
            camera_id=int(data['camera_id']),
            camera_matrix=data['camera_matrix'],
            distortion_coeffs=data['distortion_coeffs'],
            serial_number=str(data['serial_number']),
            resolution=tuple(data['resolution']),
            model=str(data['model'])
        )
        
        camera.rotation_matrix = data['rotation_matrix']
        camera.translation_vector = data['translation_vector']
        camera.is_calibrated = bool(data['is_calibrated'])
        camera.is_reference = bool(data['is_reference'])
        
        logger.info(f"Calibración de cámara {camera.camera_id} cargada desde {load_path}")
        return camera
    
    def __repr__(self):
        return (f"Camera(id={self.camera_id}, serial={self.serial_number}, "
                f"calibrated={self.is_calibrated}, reference={self.is_reference})")


class CameraSystem:
    """
    Sistema de múltiples cámaras calibradas
    """
    
    def __init__(self):
        self.cameras: Dict[int, Camera] = {}
        self.reference_camera_id: Optional[int] = None
    
    def add_camera(self, camera: Camera):
        """Añadir cámara al sistema"""
        self.cameras[camera.camera_id] = camera
        logger.info(f"Cámara {camera.camera_id} añadida al sistema")
    
    def set_reference_camera(self, camera_id: int):
        """Establecer cámara de referencia para extrínsecos"""
        if camera_id not in self.cameras:
            raise ValueError(f"Cámara {camera_id} no existe en el sistema")
        
        # Resetear referencia anterior
        if self.reference_camera_id is not None:
            old_ref = self.cameras[self.reference_camera_id]
            old_ref.is_reference = False
            old_ref.is_calibrated = False
        
        # Establecer nueva referencia
        self.reference_camera_id = camera_id
        reference_camera = self.cameras[camera_id]
        reference_camera.set_as_reference()
        
        logger.info(f"Cámara {camera_id} establecida como referencia del sistema")
    
    def initialize_from_config(self, camera_ids: List[int], reference_id: int = 0):
        """Inicializar cámaras desde configuración y establecer referencia"""
        for camera_id in camera_ids:
            camera = Camera(camera_id)
            self.add_camera(camera)
        
        # Establecer cámara de referencia
        if reference_id in camera_ids:
            self.set_reference_camera(reference_id)
        else:
            # Si no está el ID solicitado, usar el primero disponible
            self.set_reference_camera(camera_ids[0])
        
        logger.info(f"Sistema inicializado con {len(camera_ids)} cámaras, referencia: {self.reference_camera_id}")
    
    def get_camera(self, camera_id: int) -> Optional[Camera]:
        """Obtener cámara por ID"""
        return self.cameras.get(camera_id)
    
    def get_calibrated_cameras(self) -> Dict[int, Camera]:
        """Obtener solo cámaras calibradas"""
        return {id: cam for id, cam in self.cameras.items() if cam.is_calibrated}
    
    def is_system_calibrated(self) -> bool:
        """Verificar si todo el sistema está calibrado"""
        return all(cam.is_calibrated for cam in self.cameras.values())
    
    def get_projection_matrices(self) -> Dict[int, np.ndarray]:
        """Obtener matrices de proyección de todas las cámaras calibradas"""
        return {id: cam.get_projection_matrix() 
                for id, cam in self.get_calibrated_cameras().items()}
    
    def save_system(self, save_dir: Path):
        """Guardar todo el sistema de cámaras"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for camera_id, camera in self.cameras.items():
            camera.save_calibration(save_dir / f"camera_{camera_id}.npz")
        
        # Guardar metadata del sistema
        system_info = {
            'camera_ids': list(self.cameras.keys()),
            'reference_camera_id': self.reference_camera_id,
            'num_cameras': len(self.cameras)
        }
        np.savez(save_dir / "system_info.npz", **system_info)
        logger.info(f"Sistema de cámaras guardado en {save_dir}")
    
    @classmethod
    def load_system(cls, load_dir: Path) -> 'CameraSystem':
        """Cargar sistema de cámaras"""
        system = cls()
        
        # Cargar metadata
        system_info = np.load(load_dir / "system_info.npz", allow_pickle=True)
        system.reference_camera_id = int(system_info['reference_camera_id'])
        
        # Cargar cada cámara
        for camera_id in system_info['camera_ids']:
            camera_file = load_dir / f"camera_{camera_id}.npz"
            if camera_file.exists():
                camera = Camera.load_calibration(camera_file)
                system.add_camera(camera)
        
        logger.info(f"Sistema de cámaras cargado desde {load_dir}")
        return system
    
    def __len__(self):
        return len(self.cameras)
    
    def __repr__(self):
        return f"CameraSystem(cameras={len(self.cameras)}, calibrated={self.is_system_calibrated()})"

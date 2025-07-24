"""
Clase Camera para representar cámaras en el sistema de reconstrucción 3D
Encapsula parámetros intrínsecos, extrínsecos y métodos de proyección
"""
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CameraCalibrationResult:
    """Resultado de calibración de cámara"""
    success: bool
    reprojection_error: float
    calibration_points: int
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray

class Camera:
    """
    Representa una cámara en el sistema de reconstrucción 3D
    
    Atributos:
        camera_id: ID único de la cámara (0, 1, 2...)
        camera_matrix: Matriz intrínseca 3x3
        distortion_coeffs: Coeficientes de distorsión
        rotation_matrix: Matriz de rotación 3x3 (extrínsecos)
        translation_vector: Vector de traslación 3x1 (extrínsecos)
        projection_matrix: Matriz de proyección 3x4 calculada
        serial_number: Número de serie de la cámara física
    """
    
    def __init__(self, camera_id: int, serial_number: str = None):
        self.camera_id = camera_id
        self.serial_number = serial_number or f"CAMERA_{camera_id}"
        
        # Parámetros intrínsecos (se cargan desde config)
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coeffs: Optional[np.ndarray] = None
        
        # Parámetros extrínsecos (relativos a cámara de referencia)
        self.rotation_matrix: Optional[np.ndarray] = None
        self.translation_vector: Optional[np.ndarray] = None
        
        # Matriz de proyección calculada
        self.projection_matrix: Optional[np.ndarray] = None
        
        # Estado de calibración
        self.is_reference_camera = False
        self.intrinsics_calibrated = False
        self.extrinsics_calibrated = False
        
        # Cargar parámetros intrínsecos por defecto
        self._load_intrinsics()
    
    def _load_intrinsics(self):
        """Cargar parámetros intrínsecos desde configuración"""
        try:
            from config import get_camera_intrinsics
            
            self.camera_matrix, self.distortion_coeffs = get_camera_intrinsics(self.camera_id)
            self.intrinsics_calibrated = True
            
            logger.debug(f"Intrínsecos cargados para cámara {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Error cargando intrínsecos para cámara {self.camera_id}: {e}")
            self.intrinsics_calibrated = False
    
    def set_as_reference(self):
        """
        Establecer esta cámara como referencia (origen del sistema de coordenadas)
        La cámara de referencia tiene matriz identidad para rotación y vector cero para traslación
        """
        self.is_reference_camera = True
        self.rotation_matrix = np.eye(3, dtype=np.float64)
        self.translation_vector = np.zeros((3, 1), dtype=np.float64)
        self.extrinsics_calibrated = True
        
        self._update_projection_matrix()
        
        logger.info(f"📐 Cámara {self.camera_id} establecida como referencia")
    
    def set_extrinsics(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray):
        """
        Establecer parámetros extrínsecos de la cámara
        
        Args:
            rotation_matrix: Matriz de rotación 3x3
            translation_vector: Vector de traslación 3x1 o (3,)
        """
        try:
            # Validar dimensiones
            if rotation_matrix.shape != (3, 3):
                raise ValueError("rotation_matrix debe ser 3x3")
            
            # Asegurar que translation_vector sea columna
            if translation_vector.shape == (3,):
                translation_vector = translation_vector.reshape((3, 1))
            elif translation_vector.shape != (3, 1):
                raise ValueError("translation_vector debe ser (3,) o (3,1)")
            
            self.rotation_matrix = rotation_matrix.astype(np.float64)
            self.translation_vector = translation_vector.astype(np.float64)
            self.extrinsics_calibrated = True
            
            self._update_projection_matrix()
            
            logger.debug(f"Extrínsecos actualizados para cámara {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Error estableciendo extrínsecos para cámara {self.camera_id}: {e}")
            self.extrinsics_calibrated = False
    
    def _update_projection_matrix(self):
        """Calcular matriz de proyección a partir de parámetros intrínsecos y extrínsecos"""
        if not self.intrinsics_calibrated or not self.extrinsics_calibrated:
            logger.warning(f"No se puede calcular matriz de proyección para cámara {self.camera_id}")
            return
        
        try:
            # P = K * [R | t]
            # Concatenar R y t horizontalmente
            rt_matrix = np.hstack([self.rotation_matrix, self.translation_vector])  # 3x4
            
            # Multiplicar por matriz intrínseca
            self.projection_matrix = self.camera_matrix @ rt_matrix  # 3x4
            
            logger.debug(f"Matriz de proyección calculada para cámara {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Error calculando matriz de proyección para cámara {self.camera_id}: {e}")
            self.projection_matrix = None
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Proyectar puntos 3D a coordenadas 2D de imagen
        
        Args:
            points_3d: Array de puntos 3D (N, 3) o (N, 4) para coordenadas homogéneas
            
        Returns:
            Array de puntos 2D (N, 2)
        """
        if self.projection_matrix is None:
            raise ValueError(f"Matriz de proyección no disponible para cámara {self.camera_id}")
        
        try:
            # Convertir a coordenadas homogéneas si es necesario
            if points_3d.shape[1] == 3:
                # Añadir columna de unos para coordenadas homogéneas
                points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
            else:
                points_3d_homogeneous = points_3d
            
            # Proyectar: P * X = x (en coordenadas homogéneas)
            points_2d_homogeneous = (self.projection_matrix @ points_3d_homogeneous.T).T  # (N, 3)
            
            # Convertir a coordenadas cartesianas dividiendo por la tercera coordenada
            points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
            
            return points_2d
            
        except Exception as e:
            logger.error(f"Error proyectando puntos 3D a 2D para cámara {self.camera_id}: {e}")
            return np.array([])
    
    def compute_reprojection_error(self, points_3d: np.ndarray, points_2d_observed: np.ndarray) -> float:
        """
        Calcular error de reproyección entre puntos 3D proyectados y observados
        
        Args:
            points_3d: Puntos 3D (N, 3)
            points_2d_observed: Puntos 2D observados (N, 2)
            
        Returns:
            Error RMS de reproyección en píxeles
        """
        try:
            # Proyectar puntos 3D
            points_2d_projected = self.project_3d_to_2d(points_3d)
            
            if len(points_2d_projected) != len(points_2d_observed):
                raise ValueError("Número inconsistente de puntos")
            
            # Calcular diferencias
            differences = points_2d_projected - points_2d_observed
            
            # Error RMS
            squared_errors = np.sum(differences**2, axis=1)
            rms_error = np.sqrt(np.mean(squared_errors))
            
            return rms_error
            
        except Exception as e:
            logger.error(f"Error calculando reproyección para cámara {self.camera_id}: {e}")
            return float('inf')
    
    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Corregir distorsión en puntos 2D
        
        Args:
            points_2d: Puntos 2D distorsionados (N, 2)
            
        Returns:
            Puntos 2D sin distorsión (N, 2)
        """
        if not self.intrinsics_calibrated:
            logger.warning(f"Intrínsecos no calibrados para cámara {self.camera_id}")
            return points_2d
        
        try:
            # Reformatear para cv2.undistortPoints (necesita (N, 1, 2))
            points_reshaped = points_2d.reshape((-1, 1, 2)).astype(np.float32)
            
            # Corregir distorsión
            undistorted = cv2.undistortPoints(
                points_reshaped,
                self.camera_matrix,
                self.distortion_coeffs,
                P=self.camera_matrix  # Proyectar de vuelta a coordenadas de pixel
            )
            
            # Reformatear de vuelta a (N, 2)
            return undistorted.reshape((-1, 2))
            
        except Exception as e:
            logger.error(f"Error corrigiendo distorsión para cámara {self.camera_id}: {e}")
            return points_2d
    
    def calibrate_intrinsics_from_chessboard(self, chessboard_images: list, 
                                           chessboard_size: Tuple[int, int],
                                           square_size: float) -> CameraCalibrationResult:
        """
        Calibrar parámetros intrínsecos usando imágenes de tablero de ajedrez
        
        Args:
            chessboard_images: Lista de imágenes con tablero de ajedrez
            chessboard_size: Tamaño del tablero (corners_x, corners_y)
            square_size: Tamaño del cuadro en mm
            
        Returns:
            Resultado de la calibración
        """
        try:
            # Preparar puntos objeto (coordenadas 3D del tablero)
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
            objp *= square_size
            
            # Arrays para almacenar puntos objeto y imagen
            objpoints = []  # Puntos 3D en el mundo real
            imgpoints = []  # Puntos 2D en el plano de imagen
            
            for img in chessboard_images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                
                # Encontrar esquinas del tablero
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                
                if ret:
                    objpoints.append(objp)
                    
                    # Refinar esquinas
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    imgpoints.append(corners2)
            
            if len(objpoints) == 0:
                return CameraCalibrationResult(
                    success=False,
                    reprojection_error=float('inf'),
                    calibration_points=0,
                    camera_matrix=np.eye(3),
                    distortion_coeffs=np.zeros(5)
                )
            
            # Calibrar cámara
            img_size = chessboard_images[0].shape[:2][::-1]  # (width, height)
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )
            
            if ret:
                # Calcular error de reproyección
                total_error = 0
                total_points = 0
                
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(
                        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    total_error += error
                    total_points += len(imgpoints2)
                
                mean_error = total_error / len(objpoints)
                
                # Actualizar parámetros de la cámara
                self.camera_matrix = camera_matrix
                self.distortion_coeffs = dist_coeffs
                self.intrinsics_calibrated = True
                
                logger.info(f"✅ Calibración intrínseca exitosa para cámara {self.camera_id}: "
                           f"Error: {mean_error:.3f}, Puntos: {total_points}")
                
                return CameraCalibrationResult(
                    success=True,
                    reprojection_error=mean_error,
                    calibration_points=total_points,
                    camera_matrix=camera_matrix,
                    distortion_coeffs=dist_coeffs
                )
            else:
                logger.error(f"❌ Falló calibración intrínseca para cámara {self.camera_id}")
                return CameraCalibrationResult(
                    success=False,
                    reprojection_error=float('inf'),
                    calibration_points=len(objpoints),
                    camera_matrix=np.eye(3),
                    distortion_coeffs=np.zeros(5)
                )
                
        except Exception as e:
            logger.error(f"Error en calibración intrínseca para cámara {self.camera_id}: {e}")
            return CameraCalibrationResult(
                success=False,
                reprojection_error=float('inf'),
                calibration_points=0,
                camera_matrix=np.eye(3),
                distortion_coeffs=np.zeros(5)
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen del estado de la cámara"""
        return {
            'camera_id': self.camera_id,
            'serial_number': self.serial_number,
            'is_reference_camera': self.is_reference_camera,
            'intrinsics_calibrated': self.intrinsics_calibrated,
            'extrinsics_calibrated': self.extrinsics_calibrated,
            'has_projection_matrix': self.projection_matrix is not None,
            'camera_matrix_available': self.camera_matrix is not None,
            'distortion_coeffs_available': self.distortion_coeffs is not None,
            'rotation_matrix_available': self.rotation_matrix is not None,
            'translation_vector_available': self.translation_vector is not None
        }
    
    def save_parameters(self, file_path: str):
        """Guardar parámetros de la cámara en archivo"""
        try:
            params = {
                'camera_id': self.camera_id,
                'serial_number': self.serial_number,
                'camera_matrix': self.camera_matrix,
                'distortion_coeffs': self.distortion_coeffs,
                'rotation_matrix': self.rotation_matrix,
                'translation_vector': self.translation_vector,
                'is_reference_camera': self.is_reference_camera
            }
            
            np.savez(file_path, **params)
            logger.info(f"💾 Parámetros guardados para cámara {self.camera_id}: {file_path}")
            
        except Exception as e:
            logger.error(f"Error guardando parámetros para cámara {self.camera_id}: {e}")
    
    def load_parameters(self, file_path: str) -> bool:
        """
        Cargar parámetros de la cámara desde archivo
        
        Returns:
            True si se cargaron correctamente
        """
        try:
            data = np.load(file_path, allow_pickle=True)
            
            self.camera_matrix = data['camera_matrix']
            self.distortion_coeffs = data['distortion_coeffs']
            
            if 'rotation_matrix' in data and data['rotation_matrix'] is not None:
                self.rotation_matrix = data['rotation_matrix']
                self.translation_vector = data['translation_vector']
                self.extrinsics_calibrated = True
            
            if 'is_reference_camera' in data:
                self.is_reference_camera = bool(data['is_reference_camera'])
            
            self.intrinsics_calibrated = True
            self._update_projection_matrix()
            
            logger.info(f"📂 Parámetros cargados para cámara {self.camera_id}: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando parámetros para cámara {self.camera_id}: {e}")
            return False

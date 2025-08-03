import numpy as np
from typing import Dict, Tuple

# Parámetros intrínsecos por cámara (serán actualizados tras calibración)
CAMERA_INTRINSICS: Dict[str, Dict[str, np.ndarray]] = {
    # Cámara 0 - Referencia (S/N: CPE345P0007S)
    "camera_0": {
        "camera_matrix": np.array([
            [640.0, 0.0, 320.0],
            [0.0, 640.0, 240.0], 
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64),
        "serial_number": "CPE345P0007S"
    },
    
    # Cámara 1 (S/N: CPE745P0002V)
    "camera_1": {
        "camera_matrix": np.array([
            [640.0, 0.0, 320.0],
            [0.0, 640.0, 240.0],
            [0.0, 0.0, 1.0] 
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64),
        "serial_number": "CPE745P0002V"
    },
    
    # Cámara 2 (S/N: CPE745P0002B)
    "camera_2": {
        "camera_matrix": np.array([
            [640.0, 0.0, 320.0],
            [0.0, 640.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64),
        "serial_number": "CPE745P0002B"
    }
}

# Resolución de imagen de las cámaras
IMAGE_RESOLUTION = (640, 480)  # (width, height)

# Tamaño del tablero de ajedrez para calibración
CHESSBOARD_SIZE = (9, 6)  # (corners_x, corners_y)
SQUARE_SIZE = 25.0  # Tamaño del cuadro en mm

def get_camera_intrinsics(camera_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtener parámetros intrínsecos de una cámara
    
    Args:
        camera_id: ID de la cámara (0, 1, 2...)
        
    Returns:
        Tuple[camera_matrix, distortion_coeffs]
    """
    camera_key = f"camera_{camera_id}"
    
    if camera_key not in CAMERA_INTRINSICS:
        # Retornar parámetros por defecto si no existe calibración
        return get_default_intrinsics()
    
    camera_data = CAMERA_INTRINSICS[camera_key]
    return camera_data["camera_matrix"], camera_data["distortion_coeffs"]

def get_default_intrinsics() -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtener parámetros intrínsecos por defecto
    Basados en especificaciones típicas de Orbbec Gemini 335Le
    """
    # Matriz de cámara por defecto (estimación inicial)
    camera_matrix = np.array([
        [640.0, 0.0, 320.0],  # fx, 0, cx
        [0.0, 640.0, 240.0],  # 0, fy, cy  
        [0.0, 0.0, 1.0]       # 0, 0, 1
    ], dtype=np.float64)
    
    # Coeficientes de distorsión por defecto
    distortion_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    
    return camera_matrix, distortion_coeffs

def update_camera_intrinsics(camera_id: int, camera_matrix: np.ndarray, 
                           distortion_coeffs: np.ndarray, serial_number: str = None):
    """
    Actualizar parámetros intrínsecos de una cámara tras calibración
    
    Args:
        camera_id: ID de la cámara
        camera_matrix: Nueva matriz de cámara 3x3
        distortion_coeffs: Nuevos coeficientes de distorsión
        serial_number: Número de serie de la cámara (opcional)
    """
    camera_key = f"camera_{camera_id}"
    
    CAMERA_INTRINSICS[camera_key] = {
        "camera_matrix": camera_matrix.copy(),
        "distortion_coeffs": distortion_coeffs.copy(),
        "serial_number": serial_number or f"UNKNOWN_{camera_id}"
    }
    
    print(f"Parámetros intrínsecos actualizados para cámara {camera_id}")

def get_all_camera_intrinsics() -> Dict[str, Dict[str, np.ndarray]]:
    """Obtener todos los parámetros intrínsecos"""
    return CAMERA_INTRINSICS.copy()

# Parámetros para optimización de calibración
try:
    import cv2
    CALIBRATION_FLAGS = (
        cv2.CALIB_CB_ADAPTIVE_THRESH + 
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FILTER_QUADS
    )
    
    # Criterios de convergencia para calibración
    CRITERIA = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
        30, 
        0.001
    )
except ImportError:
    CALIBRATION_FLAGS = 0
    CRITERIA = None

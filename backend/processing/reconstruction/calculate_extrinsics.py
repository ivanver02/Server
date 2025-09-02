"""
Cálculo preciso de parámetros extrínsecos usando keypoints 2D.
Usa la camera0 como referencia y calcula R,t para las otras cámaras.
"""

import numpy as np
import cv2
from typing import Dict, Tuple
import logging
from backend.processing.reconstruction.camera import Camera

logger = logging.getLogger(__name__)


def calculate_extrinsics_from_keypoints(
    cameras: Dict[str, Camera],
    keypoints_2d: Dict[str, np.ndarray],
    reference_camera: str = "camera0"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Calcula parámetros extrínsecos usando correspondencias de keypoints 2D.
    
    Args:
        cameras: Dict con objetos Camera
        keypoints_2d: Dict con keypoints 2D para cada cámara (shape: Nx2)
        reference_camera: Cámara de referencia
        
    Returns:
        Dict con tuplas (R, t) para cada cámara
    """
    
    if reference_camera not in keypoints_2d:
        raise ValueError(f"Keypoints para cámara de referencia {reference_camera} no encontrados")
    
    extrinsics = {}
    ref_keypoints = keypoints_2d[reference_camera]
    ref_camera = cameras[reference_camera]
    
    # Cámara de referencia: identidad
    extrinsics[reference_camera] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
    
    for camera_id, camera in cameras.items():
        if camera_id == reference_camera:
            continue
            
        if camera_id not in keypoints_2d:
            logger.warning(f"Keypoints no encontrados para {camera_id}")
            extrinsics[camera_id] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
            continue
        
        other_keypoints = keypoints_2d[camera_id]
        
        # Filtrar correspondencias válidas (sin NaN)
        valid_mask = (~np.isnan(ref_keypoints).any(axis=1)) & (~np.isnan(other_keypoints).any(axis=1))
        
        if np.sum(valid_mask) < 8:
            logger.warning(f"Insuficientes correspondencias válidas para {camera_id}: {np.sum(valid_mask)}")
            extrinsics[camera_id] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
            continue
        
        pts_ref = ref_keypoints[valid_mask]
        pts_other = other_keypoints[valid_mask]
        
        try:
            # MÉTODO CORRECTO: Usar matriz fundamental primero
            # cv2.findFundamentalMat trabaja directamente con píxeles
            F, mask = cv2.findFundamentalMat(
                pts_ref, pts_other,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=2.0,
                confidence=0.999
            )
            
            if F is None:
                raise ValueError("No se pudo calcular matriz fundamental")
            
            # Convertir matriz fundamental a esencial: E = K2^T * F * K1
            E = camera.K.T @ F @ ref_camera.K
            
            # Recuperar pose usando puntos normalizados
            # Normalizar puntos con las matrices de cámara
            pts_ref_norm = cv2.undistortPoints(pts_ref.reshape(-1, 1, 2), ref_camera.K, None).reshape(-1, 2)
            pts_other_norm = cv2.undistortPoints(pts_other.reshape(-1, 1, 2), camera.K, None).reshape(-1, 2)
            
            inliers, R, t, mask_pose = cv2.recoverPose(E, pts_ref_norm, pts_other_norm)
            
            if inliers < 8:
                raise ValueError(f"Pocos inliers: {inliers}")
            
            # R, t representan la transformación de camera0 a camera_id
            # Para triangulación, necesitamos la transformación del mundo a cada cámara
            # Como camera0 es la referencia (mundo), estos son los extrínsecos directos
            extrinsics[camera_id] = (R.astype(np.float64), t.astype(np.float64))
            
            # Log detallado de la pose calculada
            logger.info(f"Extrínsecos calculados para {camera_id}: {inliers} inliers")
            logger.info(f"  R determinante: {np.linalg.det(R):.6f}")
            logger.info(f"  t magnitud: {np.linalg.norm(t):.3f}")
            logger.info(f"  t: {t.flatten()}")
            
        except Exception as e:
            logger.error(f"Error calculando extrínsecos para {camera_id}: {e}")
            extrinsics[camera_id] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
    
    return extrinsics

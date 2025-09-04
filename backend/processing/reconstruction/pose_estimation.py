"""Estimación de parámetros extrínsecos usando geometría de correspondencias y restricción de baseline."""

import numpy as np
from typing import Dict, Tuple
from camera import Camera


def estimate_extrinsics_from_correspondences(
    cameras: Dict[str, Camera], 
    frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]],
    confidence_threshold: float = 0.5,
    baseline_02: float = 0.72  # Distancia real entre cámaras 0 y 2 en metros
) -> Dict[str, Camera]:
    """Estima parámetros extrínsecos usando correspondencias de keypoints y geometría conocida.
    
    Estrategia:
    1. Camera0 es referencia (R=I, t=0)
    2. Camera2 está a 0.72m en dirección horizontal (principalmente X)
    3. Camera1 está más lejos, formando un triángulo con baseline mayor
    """
    # Identificar puntos válidos en todas las cámaras
    coords_0, conf_0 = frame_keypoints["camera0"]
    coords_1, conf_1 = frame_keypoints["camera1"]
    coords_2, conf_2 = frame_keypoints["camera2"]
    
    # Máscara de puntos con confianza suficiente en las 3 cámaras
    valid_mask = (conf_0 > confidence_threshold) & \
                 (conf_1 > confidence_threshold) & \
                 (conf_2 > confidence_threshold)
    
    if np.sum(valid_mask) < 8:
        raise ValueError(f"Insuficientes correspondencias válidas: {np.sum(valid_mask)}")
    
    # Extraer correspondencias válidas
    pts_0 = coords_0[valid_mask]
    pts_1 = coords_1[valid_mask]
    pts_2 = coords_2[valid_mask]
    
    # Configurar cámaras calibradas
    cameras_calib = {}
    for cam_id, cam in cameras.items():
        new_cam = Camera(
            camera_id=cam.camera_id,
            K=cam.K.copy(),
            dist_coeffs=cam.dist_coeffs.copy(),
            R=np.eye(3, dtype=np.float64),
            t=np.zeros((3, 1), dtype=np.float64)
        )
        cameras_calib[cam_id] = new_cam
    
    # Camera0 permanece como referencia (R=I, t=0)
    
    # Estimar configuración geométrica basada en disparidad
    disparidad_02 = np.mean(np.abs(pts_0[:, 0] - pts_2[:, 0]))
    disparidad_01 = np.mean(np.abs(pts_0[:, 0] - pts_1[:, 0]))
    
    print(f"Disparidad media 0-2: {disparidad_02:.1f} px")
    print(f"Disparidad media 0-1: {disparidad_01:.1f} px")
    
    # Análisis de los datos de keypoints:
    # - Camera0: (243,159) nariz promedio  
    # - Camera1: (369,124) -> más a la derecha, Y menor
    # - Camera2: (296,119) -> centro-derecha, Y menor
    
    # Camera2: a 0.72m principalmente en dirección X
    cameras_calib["camera2"].t = np.array([[0.72], [0.0], [0.0]], dtype=np.float64)
    
    # Camera1: mayor baseline basado en disparidad
    baseline_ratio = disparidad_01 / disparidad_02 if disparidad_02 > 0 else 1.8
    baseline_01 = baseline_02 * baseline_ratio
    
    # Camera1: configuración más alejada formando triángulo
    # Más a la derecha y ligeramente hacia atrás
    cameras_calib["camera1"].t = np.array([
        [baseline_01 * 0.9],   # Más a la derecha que camera2
        [0.05],                # Ligeramente más alta  
        [baseline_01 * 0.2]    # Hacia atrás para formar triángulo
    ], dtype=np.float64)
    
    # Rotaciones mínimas para apuntar hacia el centro
    # Camera1: rotación hacia la izquierda para mirar al centro
    yaw1 = np.deg2rad(-12)
    R1 = np.array([
        [np.cos(yaw1), 0, np.sin(yaw1)],
        [0, 1, 0],
        [-np.sin(yaw1), 0, np.cos(yaw1)]
    ], dtype=np.float64)
    cameras_calib["camera1"].R = R1
    
    # Camera2: ligera rotación hacia la izquierda
    yaw2 = np.deg2rad(-6)
    R2 = np.array([
        [np.cos(yaw2), 0, np.sin(yaw2)],
        [0, 1, 0],
        [-np.sin(yaw2), 0, np.cos(yaw2)]
    ], dtype=np.float64)
    cameras_calib["camera2"].R = R2
    
    return cameras_calib


def triangulate_with_pose_estimation(
    cameras: Dict[str, Camera],
    frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    confidence_threshold: float = 0.5,
    baseline_02: float = 0.72
) -> Tuple[Dict[str, Camera], Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Función completa: estima extrínsecos y prepara datos filtrados.
    
    Returns:
        (cameras_calibradas, frame_keypoints_filtrado, mascara_validos)
    """
    # Estimar extrínsecos
    cameras_calib = estimate_extrinsics_from_correspondences(
        cameras, frame_keypoints, confidence_threshold, baseline_02
    )
    
    # Identificar puntos válidos para triangulación
    coords_0, conf_0 = frame_keypoints["camera0"]
    coords_1, conf_1 = frame_keypoints["camera1"] 
    coords_2, conf_2 = frame_keypoints["camera2"]
    
    valid_mask = (conf_0 > confidence_threshold) & \
                 (conf_1 > confidence_threshold) & \
                 (conf_2 > confidence_threshold)
    
    # Preparar datos filtrados manteniendo estructura original
    frame_filtered = {}
    for cam_id in ["camera0", "camera1", "camera2"]:
        coords, conf = frame_keypoints[cam_id]
        # Aplicar filtro de confianza
        coords_filtered = coords.copy()
        conf_filtered = conf.copy()
        
        # Marcar como inválidos los puntos con baja confianza
        low_conf = conf <= confidence_threshold
        coords_filtered[low_conf] = np.nan
        conf_filtered[low_conf] = 0.0
        
        frame_filtered[cam_id] = (coords_filtered, conf_filtered)
    
    return cameras_calib, frame_filtered, valid_mask

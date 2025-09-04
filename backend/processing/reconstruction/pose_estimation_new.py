"""Estimación de parámetros extrínsecos de cámaras usando correspondencias de keypoints."""

import numpy as np
from typing import Dict, Tuple
from camera import Camera


def estimate_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Estima la matriz fundamental usando el método de 8 puntos."""
    assert pts1.shape[0] >= 8, "Se necesitan al menos 8 correspondencias"
    
    # Construir matriz A para el método de 8 puntos
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Resolver Af = 0 usando SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    
    # Hacer que F tenga rango 2
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    
    return F


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Convierte matriz fundamental a matriz esencial."""
    return K2.T @ F @ K1


def decompose_essential_matrix(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Descompone matriz esencial en las 4 posibles soluciones R,t."""
    U, _, V = np.linalg.svd(E)
    
    # Asegurar determinante positivo
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(V) < 0:
        V[-1, :] *= -1
    
    # Matriz de rotación 90 grados en Z
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Cuatro soluciones posibles
    R1 = U @ W @ V
    R2 = U @ W.T @ V
    t1 = U[:, 2].reshape(3, 1)
    t2 = -U[:, 2].reshape(3, 1)
    
    return R1, R2, t1, t2


def triangulate_point_dlt(pt1: np.ndarray, pt2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Triangula un punto usando DLT (Direct Linear Transform)."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1], 
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1]
    ])
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    return X[:3] / X[3]


def choose_correct_pose(pts1: np.ndarray, pts2: np.ndarray, K1: np.ndarray, K2: np.ndarray, 
                       R1: np.ndarray, R2: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Elige la pose correcta probando las 4 combinaciones."""
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    poses = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    best_count = -1
    best_pose = None
    
    for R, t in poses:
        P2 = K2 @ np.hstack([R, t])
        count_positive = 0
        
        for i in range(min(len(pts1), 10)):  # Probar con los primeros 10 puntos
            pt3d = triangulate_point_dlt(pts1[i], pts2[i], P1, P2)
            
            # Verificar que el punto esté delante de ambas cámaras
            z1 = pt3d[2]
            pt3d_cam2 = R @ pt3d + t.flatten()
            z2 = pt3d_cam2[2]
            
            if z1 > 0 and z2 > 0:
                count_positive += 1
        
        if count_positive > best_count:
            best_count = count_positive
            best_pose = (R, t)
    
    return best_pose


def estimate_extrinsics_algebraic(
    cameras: Dict[str, Camera], 
    frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]],
    confidence_threshold: float = 0.5,
    baseline_02: float = 0.72
) -> Dict[str, Camera]:
    """Estima parámetros extrínsecos usando álgebra de múltiples vistas."""
    
    # Identificar puntos válidos
    coords_0, conf_0 = frame_keypoints["camera0"]
    coords_1, conf_1 = frame_keypoints["camera1"]
    coords_2, conf_2 = frame_keypoints["camera2"]
    
    valid_mask = (conf_0 > confidence_threshold) & \
                 (conf_1 > confidence_threshold) & \
                 (conf_2 > confidence_threshold)
    
    if np.sum(valid_mask) < 8:
        raise ValueError(f"Insuficientes correspondencias válidas: {np.sum(valid_mask)}")
    
    pts_0 = coords_0[valid_mask]
    pts_1 = coords_1[valid_mask]
    pts_2 = coords_2[valid_mask]
    
    # Configurar cámara 0 como referencia
    cameras_calib = cameras.copy()
    cameras_calib["camera0"].R = np.eye(3, dtype=np.float64)
    cameras_calib["camera0"].t = np.zeros((3, 1), dtype=np.float64)
    
    K0 = cameras["camera0"].K
    K1 = cameras["camera1"].K
    K2 = cameras["camera2"].K
    
    # Estimar pose para cámara 2 (sabemos que está a 72cm)
    F_02 = estimate_fundamental_matrix(pts_0, pts_2)
    E_02 = essential_from_fundamental(F_02, K0, K2)
    
    R1_02, R2_02, t1_02, t2_02 = decompose_essential_matrix(E_02)
    R_02, t_02 = choose_correct_pose(pts_0, pts_2, K0, K2, R1_02, R2_02, t1_02, t2_02)
    
    # Escalar usando la distancia conocida
    scale_factor = baseline_02 / np.linalg.norm(t_02)
    t_02_scaled = t_02 * scale_factor
    
    cameras_calib["camera2"].R = R_02.astype(np.float64)
    cameras_calib["camera2"].t = t_02_scaled.astype(np.float64)
    
    # Estimar pose para cámara 1
    F_01 = estimate_fundamental_matrix(pts_0, pts_1)
    E_01 = essential_from_fundamental(F_01, K0, K1)
    
    R1_01, R2_01, t1_01, t2_01 = decompose_essential_matrix(E_01)
    R_01, t_01 = choose_correct_pose(pts_0, pts_1, K0, K1, R1_01, R2_01, t1_01, t2_01)
    
    # Escalar la traslación de cámara 1 usando la misma escala
    t_01_scaled = t_01 * scale_factor
    
    cameras_calib["camera1"].R = R_01.astype(np.float64)
    cameras_calib["camera1"].t = t_01_scaled.astype(np.float64)
    
    return cameras_calib


def triangulate_with_pose_estimation(
    cameras: Dict[str, Camera],
    frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    confidence_threshold: float = 0.5,
    baseline_02: float = 0.72
) -> Tuple[Dict[str, Camera], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Función completa: estima extrínsecos y prepara datos para triangulación."""
    
    # Estimar extrínsecos
    cameras_calib = estimate_extrinsics_algebraic(
        cameras, frame_keypoints, confidence_threshold, baseline_02
    )
    
    # Identificar puntos válidos
    coords_0, conf_0 = frame_keypoints["camera0"]
    coords_1, conf_1 = frame_keypoints["camera1"] 
    coords_2, conf_2 = frame_keypoints["camera2"]
    
    valid_mask = (conf_0 > confidence_threshold) & \
                 (conf_1 > confidence_threshold) & \
                 (conf_2 > confidence_threshold)
    
    # Crear estructura de datos filtrada
    frame_filtered = {}
    for cam_id in ["camera0", "camera1", "camera2"]:
        coords, conf = frame_keypoints[cam_id]
        # Marcar como NaN los puntos inválidos
        coords_filtered = coords.copy()
        coords_filtered[~valid_mask] = np.nan
        conf_filtered = conf.copy()
        conf_filtered[~valid_mask] = 0.0
        
        frame_filtered[cam_id] = (coords_filtered, conf_filtered)
    
    return cameras_calib, frame_filtered

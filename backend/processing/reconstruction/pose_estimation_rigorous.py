"""Estimación de extrínsecos usando geometría epipolar rigurosa."""

import numpy as np
from typing import Dict, Tuple, List
from camera import Camera


def estimate_fundamental_matrix_ransac(pts1: np.ndarray, pts2: np.ndarray, iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Estima matriz fundamental usando RANSAC para robustez."""
    
    def compute_fundamental_8point(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Método de 8 puntos para matriz fundamental."""
        A = np.zeros((len(p1), 9))
        for i in range(len(p1)):
            x1, y1 = p1[i]
            x2, y2 = p2[i]
            A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        
        _, _, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)
        
        # Enforcer rango 2
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ V
        return F
    
    def compute_epipolar_error(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calcula error epipolar para cada correspondencia."""
        errors = []
        for i in range(len(p1)):
            x1 = np.array([p1[i, 0], p1[i, 1], 1])
            x2 = np.array([p2[i, 0], p2[i, 1], 1])
            
            # Error simétrico
            err1 = np.abs(x2.T @ F @ x1)
            line1 = F @ x1
            line2 = F.T @ x2
            
            err1 /= np.sqrt(line1[0]**2 + line1[1]**2)
            err2 = np.abs(x1.T @ F.T @ x2) / np.sqrt(line2[0]**2 + line2[1]**2)
            
            errors.append((err1 + err2) / 2)
        return np.array(errors)
    
    best_F = None
    best_inliers = None
    best_score = 0
    threshold = 3.0  # píxeles
    
    for _ in range(iterations):
        # Seleccionar 8 puntos aleatorios
        indices = np.random.choice(len(pts1), 8, replace=False)
        sample1, sample2 = pts1[indices], pts2[indices]
        
        try:
            F_candidate = compute_fundamental_8point(sample1, sample2)
            errors = compute_epipolar_error(F_candidate, pts1, pts2)
            inliers = errors < threshold
            score = np.sum(inliers)
            
            if score > best_score:
                best_score = score
                best_F = F_candidate
                best_inliers = inliers
                
        except:
            continue
    
    return best_F, best_inliers


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Convierte matriz fundamental a esencial."""
    return K2.T @ F @ K1


def decompose_essential_matrix(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Descompone matriz esencial en las 4 soluciones posibles."""
    U, _, Vt = np.linalg.svd(E)
    
    # Asegurar determinantes correctos
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    
    solutions = [
        (R1, t.reshape(3, 1)),
        (R1, -t.reshape(3, 1)),
        (R2, t.reshape(3, 1)),
        (R2, -t.reshape(3, 1))
    ]
    
    return solutions


def triangulate_points_linear(pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Triangulación lineal usando DLT."""
    points_3d = []
    
    for i in range(len(pts1)):
        A = np.array([
            pts1[i, 0] * P1[2] - P1[0],
            pts1[i, 1] * P1[2] - P1[1],
            pts2[i, 0] * P2[2] - P2[0],
            pts2[i, 1] * P2[2] - P2[1]
        ])
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Homogéneas a euclidiano
        points_3d.append(X[:3])
    
    return np.array(points_3d)


def count_points_in_front(pts_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> int:
    """Cuenta puntos que están delante de ambas cámaras."""
    count = 0
    
    for pt in pts_3d:
        # Camera 1 (identidad)
        if pt[2] > 0:
            # Camera 2
            pt_cam2 = R @ pt + t.flatten()
            if pt_cam2[2] > 0:
                count += 1
    
    return count


def estimate_extrinsics_rigorous(
    cameras: Dict[str, Camera],
    frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]],
    confidence_threshold: float = 0.5,
    baseline_02: float = 0.72
) -> Dict[str, Camera]:
    """Estimación rigurosa usando geometría epipolar."""
    
    # Extraer puntos válidos
    coords_0, conf_0 = frame_keypoints["camera0"]
    coords_1, conf_1 = frame_keypoints["camera1"]
    coords_2, conf_2 = frame_keypoints["camera2"]
    
    valid_mask = (conf_0 > confidence_threshold) & \
                 (conf_1 > confidence_threshold) & \
                 (conf_2 > confidence_threshold)
    
    if np.sum(valid_mask) < 8:
        raise ValueError(f"Insuficientes correspondencias: {np.sum(valid_mask)}")
    
    pts_0 = coords_0[valid_mask]
    pts_1 = coords_1[valid_mask] 
    pts_2 = coords_2[valid_mask]
    
    print(f"Usando {len(pts_0)} correspondencias para estimación rigurosa")
    
    # Copiar cámaras
    cameras_calib = {}
    for cam_id, cam in cameras.items():
        cameras_calib[cam_id] = Camera(
            camera_id=cam.camera_id,
            K=cam.K.copy(),
            dist_coeffs=cam.dist_coeffs.copy(),
            R=np.eye(3, dtype=np.float64),
            t=np.zeros((3, 1), dtype=np.float64)
        )
    
    # Camera0 es referencia
    K0, K1, K2 = cameras["camera0"].K, cameras["camera1"].K, cameras["camera2"].K
    
    # Estimar para par 0-2 (baseline conocido)
    print("Estimando extrínsecos 0-2...")
    F_02, inliers_02 = estimate_fundamental_matrix_ransac(pts_0, pts_2)
    E_02 = essential_from_fundamental(F_02, K0, K2)
    solutions_02 = decompose_essential_matrix(E_02)
    
    # Evaluar soluciones por triangulación
    best_R_02, best_t_02 = None, None
    best_count_02 = 0
    
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    for R, t in solutions_02:
        P2 = K2 @ np.hstack([R, t])
        pts_3d = triangulate_points_linear(pts_0[inliers_02], pts_2[inliers_02], P0, P2)
        count = count_points_in_front(pts_3d, R, t)
        
        if count > best_count_02:
            best_count_02 = count
            best_R_02, best_t_02 = R, t
    
    # Escalar con baseline conocido
    scale_02 = baseline_02 / np.linalg.norm(best_t_02)
    best_t_02 *= scale_02
    
    cameras_calib["camera2"].R = best_R_02
    cameras_calib["camera2"].t = best_t_02
    
    print(f"Camera2: {best_count_02} puntos delante, scale={scale_02:.3f}")
    
    # Similar para par 0-1
    print("Estimando extrínsecos 0-1...")
    F_01, inliers_01 = estimate_fundamental_matrix_ransac(pts_0, pts_1)
    E_01 = essential_from_fundamental(F_01, K0, K1)
    solutions_01 = decompose_essential_matrix(E_01)
    
    best_R_01, best_t_01 = None, None
    best_count_01 = 0
    
    for R, t in solutions_01:
        P1 = K1 @ np.hstack([R, t])
        pts_3d = triangulate_points_linear(pts_0[inliers_01], pts_1[inliers_01], P0, P1)
        count = count_points_in_front(pts_3d, R, t)
        
        if count > best_count_01:
            best_count_01 = count
            best_R_01, best_t_01 = R, t
    
    # Escalar usando relación de baseline estimada
    scale_01 = scale_02 * np.linalg.norm(best_t_01) / np.linalg.norm(best_t_02) * 1.5  # Factor empírico
    best_t_01 = best_t_01 / np.linalg.norm(best_t_01) * (baseline_02 * 1.8)  # Baseline mayor
    
    cameras_calib["camera1"].R = best_R_01
    cameras_calib["camera1"].t = best_t_01
    
    print(f"Camera1: {best_count_01} puntos delante, baseline={np.linalg.norm(best_t_01):.3f}m")
    
    return cameras_calib

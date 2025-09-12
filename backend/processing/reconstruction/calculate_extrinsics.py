import numpy as np
from typing import Dict, Tuple, List

try:
    from .camera import Camera
except ImportError:
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


def estimate_extrinsics(
    cameras: Dict[str, Camera],
    frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]],
    confidence_threshold: float = 0.5,
    reference_camera: str = "camera0"
) -> Dict[str, Camera]:
    
    camera_ids = list(cameras.keys())
    
    # Validar que la cámara de referencia existe
    if reference_camera not in camera_ids:
        raise ValueError(f"Cámara de referencia {reference_camera} no encontrada en {camera_ids}")
    
    # Extraer todos los puntos y crear máscara de validez
    all_coords = {}
    all_confs = {}
    
    for cam_id in camera_ids:
        coords, confs = frame_keypoints[cam_id]
        all_coords[cam_id] = coords
        all_confs[cam_id] = confs
    
    # Crear máscara de puntos válidos en todas las cámaras
    valid_mask = all_confs[camera_ids[0]] > confidence_threshold
    for cam_id in camera_ids[1:]:
        valid_mask &= all_confs[cam_id] > confidence_threshold
    
    if np.sum(valid_mask) < 8:
        raise ValueError(f"Insuficientes correspondencias: {np.sum(valid_mask)}")
    
    # Extraer puntos válidos para todas las cámaras
    valid_points = {}
    for cam_id in camera_ids:
        valid_points[cam_id] = all_coords[cam_id][valid_mask]
    
    print(f"Usando {len(valid_points[camera_ids[0]])} correspondencias para estimación")
    
    # Copiar cámaras inicializando con pose identidad
    cameras_calib = {}
    for cam_id, cam in cameras.items():
        cameras_calib[cam_id] = Camera(
            camera_id=cam.camera_id,
            K=cam.K.copy(),
            dist_coeffs=cam.dist_coeffs.copy(),
            R=np.eye(3, dtype=np.float64),
            t=np.zeros((3, 1), dtype=np.float64)
        )
    
    # Matriz de proyección de la cámara de referencia (identidad)
    K_ref = cameras[reference_camera].K
    P_ref = K_ref @ np.hstack([np.eye(3), np.zeros((3, 1))])
    pts_ref = valid_points[reference_camera]
    
    # Estimar extrínsecos para todas las demás cámaras respecto a la de referencia
    for cam_id in camera_ids:
        if cam_id == reference_camera:
            continue  # Cámara de referencia mantiene pose identidad
            
        print(f"Estimando extrínsecos {reference_camera}-{cam_id}...")
        
        pts_cam = valid_points[cam_id]
        K_cam = cameras[cam_id].K
        
        # Estimar matriz fundamental
        F, inliers = estimate_fundamental_matrix_ransac(pts_ref, pts_cam)
        E = essential_from_fundamental(F, K_ref, K_cam)
        solutions = decompose_essential_matrix(E)
        
        # Evaluar soluciones por triangulación
        best_R, best_t = None, None
        best_count = 0
        
        for R, t in solutions:
            P_cam = K_cam @ np.hstack([R, t])
            pts_3d = triangulate_points_linear(pts_ref[inliers], pts_cam[inliers], P_ref, P_cam)
            count = count_points_in_front(pts_3d, R, t)
            
            if count > best_count:
                best_count = count
                best_R, best_t = R, t
        
        # Asignar mejor solución
        cameras_calib[cam_id].R = best_R
        cameras_calib[cam_id].t = best_t
        
        baseline = np.linalg.norm(best_t)
        print(f"{cam_id}: {best_count} puntos delante, baseline={baseline:.3f} (escala natural)")
    
    return cameras_calib


def print_extrinsic_matrices(cameras: Dict[str, Camera], title: str = "PARÁMETROS EXTRÍNSECOS"):
    """
    Muestra las matrices de parámetros extrínsecos (R, t) para todas las cámaras.
    
    Args:
        cameras: Diccionario de cámaras con parámetros extrínsecos
        title: Título a mostrar en el encabezado
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    camera_ids = sorted(cameras.keys())
    
    for cam_id in camera_ids:
        cam = cameras[cam_id]
        print(f"\n{cam_id.upper()}:")
        
        if hasattr(cam, 'R') and hasattr(cam, 't'):
            print("  Matriz de Rotación (R):")
            for i, row in enumerate(cam.R):
                print(f"    [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")
            
            print("  Vector de Traslación (t):")
            t_flat = cam.t.flatten()
            print(f"    [{t_flat[0]:8.5f}, {t_flat[1]:8.5f}, {t_flat[2]:8.5f}] metros")
            
            # Información adicional
            baseline = np.linalg.norm(cam.t)
            print(f"  Baseline desde origen: {baseline:.5f} metros")
            
            # Ángulos de Euler aproximados
            import math
            rx = math.degrees(math.atan2(cam.R[2,1], cam.R[2,2]))
            ry = math.degrees(math.atan2(-cam.R[2,0], math.sqrt(cam.R[2,1]**2 + cam.R[2,2]**2)))
            rz = math.degrees(math.atan2(cam.R[1,0], cam.R[0,0]))
            print(f"  Ángulos aprox: Rx={rx:.2f}°, Ry={ry:.2f}°, Rz={rz:.2f}°")
        else:
            print("  Sin parámetros extrínsecos (cámara de referencia)")
    
    print(f"\n{'='*70}")

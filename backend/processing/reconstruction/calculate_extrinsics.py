"""Cálculo preciso de extrínsecos relativos a camera0 usando sólo keypoints 2D.

Algoritmo:
1. Detectar cámaras disponibles (directorios data/processed/2D_keypoints/<paciente>/<sesion>/cameraX)
2. Cargar todos los frames del chunk indicado (por requisito: chunk 0 en test)
3. Reunir correspondencias para cada cámara vs camera0: (x,y) por keypoint y frame con confianza válida
4. Normalizar usando K^{-1} de cada cámara para obtener coordenadas en el plano imagen normalizado
5. Estimar matriz esencial E mediante algoritmo de 8 puntos normalizado + RANSAC ligero propio:
   - Se generan muestras aleatorias de 8 correspondencias
   - Se evalúa con error epipolar |x2^T E x1| / sqrt((Ex1)_0^2 + (Ex1)_1^2)
6. Refinar E con todas las inliers y forzar rango 2
7. Descomponer E en (R, t) candidatos, elegir el que maximiza puntos triangulados delante de ambas cámaras
8. Guardar R, t en las instancias Camera (camera0 identidad)

Nota: La escala de t queda arbitraria (estructura sólo definida hasta escala). Puede fijarse a unidad.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Dict, List, Tuple
from .camera import Camera


def _list_cameras(base_path: str) -> List[str]:
    return sorted([d for d in os.listdir(base_path) if d.startswith("camera") and os.path.isdir(os.path.join(base_path, d))])


def _load_frame_ids(camera_dir: str, chunk_id: int) -> List[int]:
    coord_dir = os.path.join(camera_dir, "coordinates")
    if not os.path.isdir(coord_dir):
        return []
    frame_ids = []
    suffix = f"_{chunk_id}.npy"
    for fname in os.listdir(coord_dir):
        if fname.endswith(suffix):
            try:
                frame_id = int(fname.split("_")[0])
                frame_ids.append(frame_id)
            except ValueError:
                pass
    return sorted(frame_ids)


def _load_keypoint_arrays(camera_path: str, frame_id: int, chunk_id: int) -> Tuple[np.ndarray, np.ndarray]:
    c_path = os.path.join(camera_path, "coordinates", f"{frame_id}_{chunk_id}.npy")
    conf_path = os.path.join(camera_path, "confidence", f"{frame_id}_{chunk_id}.npy")
    coords = np.load(c_path)  # (K,2)
    conf = np.load(conf_path)  # (K,)
    return coords, conf


def _eight_point_E(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    # points*: (N,2) normalized
    N = points1.shape[0]
    A = np.zeros((N, 9))
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0], points2[:, 1]
    A[:, 0] = x2 * x1
    A[:, 1] = x2 * y1
    A[:, 2] = x2
    A[:, 3] = y2 * x1
    A[:, 4] = y2 * y1
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1
    A[:, 8] = 1.0
    # Solve via SVD
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    # Enforce rank 2
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0.0
    E = U @ np.diag(S) @ Vt
    return E / np.linalg.norm(E)


def _score_E(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, thresh: float) -> Tuple[np.ndarray, float]:
    # Sampson distance
    ones = np.ones((pts1.shape[0], 1))
    x1h = np.hstack([pts1, ones])
    x2h = np.hstack([pts2, ones])
    Ex1 = (E @ x1h.T).T
    Etx2 = (E.T @ x2h.T).T
    x2tEx1 = np.sum(x2h * Ex1, axis=1)
    denom = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    dist = (x2tEx1 ** 2) / denom
    inliers = dist < thresh
    score = inliers.sum()
    return inliers, score


def _estimate_E_ransac(norm1: np.ndarray, norm2: np.ndarray, iterations: int = 2000, thresh: float = 1e-4) -> np.ndarray:
    best_E = None
    best_inliers = None
    n = norm1.shape[0]
    if n < 8:
        raise ValueError("Se requieren al menos 8 correspondencias para estimar E")
    idx = np.arange(n)
    for _ in range(iterations):
        sample = np.random.choice(idx, 8, replace=False)
        E_candidate = _eight_point_E(norm1[sample], norm2[sample])
        inliers, score = _score_E(E_candidate, norm1, norm2, thresh)
        if best_inliers is None or score > best_inliers.sum():
            best_E = E_candidate
            best_inliers = inliers
    # Recompute with all inliers
    E_refined = _eight_point_E(norm1[best_inliers], norm2[best_inliers])
    return E_refined


def _decompose_E(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def _triangulate_point(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    A = np.zeros((4, 4))
    A[0] = x1[0] * P1[2] - P1[0]
    A[1] = x1[1] * P1[2] - P1[1]
    A[2] = x2[0] * P2[2] - P2[0]
    A[3] = x2[1] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]
    return X[:3]


def _choose_pose(candidates, K1, K2, pts1, pts2):
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    best_count = -1
    for R, t in candidates:
        P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
        count = 0
        for i in range(min(pts1.shape[0], 50)):
            X = _triangulate_point(P1, P2, pts1[i], pts2[i])
            # Cheirality: delante de ambas cámaras
            if X[2] > 0 and (R @ X + t)[2] > 0:
                count += 1
        if count > best_count:
            best = (R, t)
            best_count = count
    return best


def calculate_extrinsics(patient_id: str, session_id: str, chunk_id: int = 0, base_data_dir: str = "data/processed/2D_keypoints") -> Dict[str, Camera]:
    session_path = os.path.join(base_data_dir, patient_id, session_id)
    cameras_ids = _list_cameras(session_path)
    cameras: Dict[str, Camera] = {cid: Camera.create(cid) for cid in cameras_ids}
    if "camera0" not in cameras:
        raise RuntimeError("Se requiere camera0 como referencia")

    # Reunir correspondencias camera0 vs cada otra
    cam0_dir = os.path.join(session_path, "camera0")
    frame_ids = _load_frame_ids(cam0_dir, chunk_id)
    if not frame_ids:
        raise RuntimeError("No se encontraron frames para camera0")

    K0 = cameras["camera0"].K
    K0_inv = np.linalg.inv(K0)

    for cid, cam in cameras.items():
        if cid == "camera0":
            continue
        c_dir = os.path.join(session_path, cid)
        Kc = cam.K
        Kc_inv = np.linalg.inv(Kc)
        pts0_norm = []
        ptsc_norm = []
        for fid in frame_ids:
            try:
                k0, conf0 = _load_keypoint_arrays(os.path.join(session_path, "camera0"), fid, chunk_id)
                kc, confc = _load_keypoint_arrays(c_dir, fid, chunk_id)
            except FileNotFoundError:
                continue
            valid = (conf0 > 0) & (confc > 0)
            if not np.any(valid):
                continue
            p0 = k0[valid]
            pc = kc[valid]
            # Normalizar
            p0_h = np.hstack([p0, np.ones((p0.shape[0], 1))])
            pc_h = np.hstack([pc, np.ones((pc.shape[0], 1))])
            p0_n = (K0_inv @ p0_h.T).T[:, :2]
            pc_n = (Kc_inv @ pc_h.T).T[:, :2]
            pts0_norm.append(p0_n)
            ptsc_norm.append(pc_n)
        if not pts0_norm:
            raise RuntimeError(f"No hay correspondencias válidas entre camera0 y {cid}")
        pts0_norm = np.vstack(pts0_norm)
        ptsc_norm = np.vstack(ptsc_norm)
        # Estimar E
        E = _estimate_E_ransac(pts0_norm, ptsc_norm)
        candidates = _decompose_E(E)
        R, t = _choose_pose(candidates, K0, cam.K, pts0_norm, ptsc_norm)
        # Normalizar t (escala unitaria)
        t = t / np.linalg.norm(t)
        cam.R = R
        cam.t = t.reshape(3, 1)
    return cameras
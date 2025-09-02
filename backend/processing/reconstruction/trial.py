"""Prueba sintética de reconstrucción 3D.

Genera:
 - Puntos 3D arbitrarios (simulan esqueleto sencillo)
 - 3 cámaras con intrínsecos y extrínsecos conocidos
 - Proyecciones 2D (formato similar a processed: (K,2) + (K,) confidence)

Luego:
 - Triangula con SVD
 - Refinamiento con Bundle Adjustment
 - Calcula error de reproyección (px) y error 3D respecto al ground truth

Objetivo: verificar que con datos limpios el pipeline produce error casi nulo.
"""
from __future__ import annotations

import numpy as np
from camera import Camera
from triangulation_svd import triangulate_frame_svd
from triangulation_bundle_adjustment import refine_frame_bundle_adjustment
from reprojection import reprojection_error


def generate_ground_truth_3d() -> np.ndarray:
    # 17 puntos estilo COCO simplificado (arbitrarios en torno a z≈2m)
    pts = np.array([
        [0.0, 1.70, 2.5],   # Nose
        [-0.15, 1.55, 2.5], # Eye L
        [0.15, 1.55, 2.5],  # Eye R
        [-0.25, 1.35, 2.5], # Shoulder L
        [0.25, 1.35, 2.5],  # Shoulder R
        [-0.30, 1.00, 2.5], # Elbow L
        [0.30, 1.00, 2.5],  # Elbow R
        [-0.35, 0.70, 2.55],# Wrist L
        [0.35, 0.70, 2.55], # Wrist R
        [0.0, 1.10, 2.5],   # Mid torso
        [-0.15, 0.90, 2.55],# Hip L
        [0.15, 0.90, 2.55], # Hip R
        [-0.15, 0.50, 2.65],# Knee L
        [0.15, 0.50, 2.65], # Knee R
        [-0.15, 0.10, 2.75],# Ankle L
        [0.15, 0.10, 2.75], # Ankle R
        [0.0, 0.00, 2.80],  # Foot mid
    ], dtype=np.float64)
    return pts


def create_cameras():
    # Usar intrínsecos reales desde config (camera_intrinsics.py) vía Camera.create
    cams = {cid: Camera.create(cid) for cid in ["camera0", "camera1", "camera2"]}

    # Extrínsecos sintéticos: camera0 referencia
    cams["camera0"].R = np.eye(3)
    cams["camera0"].t = np.zeros((3, 1))

    # Camera1: desplazada 0.5m a la derecha y rotada -8° yaw
    yaw1 = np.deg2rad(-8)
    R1 = np.array([
        [np.cos(yaw1), 0, np.sin(yaw1)],
        [0, 1, 0],
        [-np.sin(yaw1), 0, np.cos(yaw1)],
    ])
    t1 = np.array([[0.5], [0.0], [0.0]])
    cams["camera1"].R = R1
    cams["camera1"].t = t1

    # Camera2: desplazada 0.6m a la izquierda y rotada +10° yaw
    yaw2 = np.deg2rad(10)
    R2 = np.array([
        [np.cos(yaw2), 0, np.sin(yaw2)],
        [0, 1, 0],
        [-np.sin(yaw2), 0, np.cos(yaw2)],
    ])
    t2 = np.array([[-0.6], [0.0], [0.05]])
    cams["camera2"].R = R2
    cams["camera2"].t = t2

    return cams


def project_points(cams, pts3d, noise_px=0.0):
    frame_keypoints = {}
    for cid, cam in cams.items():
        pts2d = cam.project(pts3d)
        if noise_px > 0:
            pts2d += np.random.normal(0, noise_px, pts2d.shape)
        conf = np.ones((pts2d.shape[0],), dtype=np.float64)
        frame_keypoints[cid] = (pts2d, conf)
    return frame_keypoints


def compute_3d_error(gt, pred):
    valid = ~np.isnan(pred[:, 0])
    if not np.any(valid):
        return np.nan
    diff = gt[valid] - pred[valid]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def main():
    np.random.seed(42)
    pts3d_gt = generate_ground_truth_3d()
    cams = create_cameras()
    frame = project_points(cams, pts3d_gt, noise_px=0.0)

    print("Triangulación SVD...")
    pts3d_svd = triangulate_frame_svd(cams, frame)
    err3d_svd = compute_3d_error(pts3d_gt, pts3d_svd)
    reproj_svd = reprojection_error(pts3d_svd, cams, frame)
    print(f"Error 3D medio SVD: {err3d_svd:.6f} m")
    for cid, e in reproj_svd.items():
        print(f"  Reproj {cid}: {e:.6f} px")

    print("\nRefinamiento Bundle Adjustment...")
    pts3d_ba = refine_frame_bundle_adjustment(pts3d_svd, cams, frame)
    err3d_ba = compute_3d_error(pts3d_gt, pts3d_ba)
    reproj_ba = reprojection_error(pts3d_ba, cams, frame)
    print(f"Error 3D medio BA:  {err3d_ba:.6f} m")
    for cid, e in reproj_ba.items():
        print(f"  Reproj {cid}: {e:.6f} px")

    # Diferencia entre SVD y BA
    if not np.isnan(err3d_svd) and not np.isnan(err3d_ba):
        print(f"\nMejora BA (Δ error 3D): {err3d_svd - err3d_ba:.6e} m")

    # Comprobación rápida de que los errores de reproyección son ~0
    max_reproj_svd = np.nanmax(list(reproj_svd.values()))
    max_reproj_ba = np.nanmax(list(reproj_ba.values()))
    print(f"Max reproj SVD: {max_reproj_svd:.6e} px | Max reproj BA: {max_reproj_ba:.6e} px")


if __name__ == "__main__":
    main()
from __future__ import annotations

import numpy as np
from camera import Camera
from triangulation_svd import triangulate_frame_svd
from triangulation_bundle_adjustment import refine_frame_bundle_adjustment
from reprojection import reprojection_error
from pathlib import Path
from typing import Tuple

coordinates_camera_0 = np.array([[243.50415277, 159.37580359],
       [248.05017458, 155.09014137],
       [238.51039355, 153.8757195 ],
       [252.39126702, 158.54118418],
       [227.87675844, 155.12744863],
       [257.48707717, 191.7911766 ],
       [212.89595865, 184.2994381 ],
       [263.61332671, 230.39549047],
       [176.95070811, 202.42121691],
       [269.57449521, 264.37477131],
       [184.74043683, 162.4216488 ],
       [249.67777976, 268.03960046],
       [219.16895593, 267.13665592],
       [245.97380636, 334.52112875],
       [215.30345628, 335.72896562],
       [241.18540775, 401.32784923],
       [213.96846082, 404.12025901],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ]])

confidences_camera_0 = np.array([0.96747363, 0.9862318 , 0.95895183, 0.96982431, 0.97488534,
       0.93971992, 0.93201089, 0.98253667, 0.96078122, 0.97526371,
       0.96469021, 0.91449094, 0.88579488, 0.95138872, 0.93920207,
       0.90330499, 0.91930181, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        ])

coordinates_camera_1 = np.array([[369.17324858, 124.11412505],
       [373.60018341, 118.95813442],
       [364.9677895 , 119.95915198],
       [383.82330943, 120.36893765],
       [361.02404947, 123.1058976 ],
       [398.52998161, 151.80352073],
       [353.16337364, 154.13660596],
       [410.69604813, 192.95295275],
       [320.05695356, 174.51739911],
       [407.2497562 , 227.87381408],
       [314.92151524, 141.09700972],
       [385.06408534, 226.26452067],
       [355.49175552, 226.30814812],
       [385.70701632, 290.5136311 ],
       [360.07108841, 289.24549006],
       [386.38307456, 352.65304715],
       [364.95822034, 348.23664068],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ]])

confidences_camera_1 = np.array([0.96523976, 0.98087442, 0.99132872, 0.97956949, 0.94814324,
       0.9478066 , 0.94084626, 0.9795531 , 0.94840074, 0.96928489,
       0.96984708, 0.88265347, 0.87077898, 0.92599779, 0.95448089,
       0.91450763, 0.91984427, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        ])

coordinates_camera_2 = np.array([[296.07968708, 119.44613515],
       [300.23493034, 114.25656826],
       [290.89613374, 114.84839878],
       [307.35056466, 116.55150774],
       [284.26480287, 118.13758883],
       [320.9347374 , 149.38203238],
       [273.37982928, 149.69611317],
       [332.74144146, 190.13976467],
       [241.26046484, 171.89268154],
       [336.49152444, 224.68489952],
       [239.42382249, 132.49654104],
       [314.96261597, 224.8332871 ],
       [284.73677668, 225.93263434],
       [317.34340542, 290.06601704],
       [288.85827279, 290.86280624],
       [319.48270232, 351.58898213],
       [294.85371359, 352.20091297],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ],
       [  0.        ,   0.        ]])

confidences_camera_2 = np.array([0.97985238, 0.98296541, 0.98628592, 0.98535669, 0.97465611,
       0.94587588, 0.95796943, 0.94574189, 0.96294028, 0.94961226,
       0.99918878, 0.86759275, 0.88666171, 0.95032251, 0.95671833,
       0.93984115, 0.91424793, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        ])


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


def load_keypoint_files(patient: int | str, session: int | str, camera: int | str, frame: int, chunk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Carga los archivos .npy de coordenadas y confianza para un frame específico.

    Parámetros:
        patient: número o string del paciente (ej: 2 -> 'patient2')
        session: número o string de la sesión (ej: 1 -> 'session1')
        camera: índice o string de la cámara (ej: 0 -> 'camera0')
        frame: número de frame (ej: 150)
        chunk: número de chunk (ej: 0)

    Devuelve:
        (coordinates, confidence) como arrays numpy.
        Lanza FileNotFoundError si falta el de coordenadas.
        Si falta el de confianza, devuelve un array de unos del mismo tamaño.
    """
    root = Path(__file__).resolve().parents[3]  # Directorio 'Server'
    base = root / "data" / "processed" / "2D_keypoints" / f"patient{patient}" / f"session{session}" / f"camera{camera}"
    coord_path = base / "coordinates" / f"{frame}_{chunk}.npy"
    conf_path = base / "confidence" / f"{frame}_{chunk}.npy"
    if not coord_path.exists():
        raise FileNotFoundError(f"No existe archivo de coordenadas: {coord_path}")
    coordinates = np.load(coord_path).astype(np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Formato inválido en {coord_path}, se esperaba (K,2)")
    if conf_path.exists():
        confidence = np.load(conf_path).astype(np.float64)
        if confidence.shape[0] != coordinates.shape[0]:
            raise ValueError(f"Confianza tamaño {confidence.shape[0]} != coordenadas {coordinates.shape[0]} en {conf_path}")
    else:
        confidence = np.ones((coordinates.shape[0],), dtype=np.float64)
    return coordinates, confidence


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
    print(load_keypoint_files(2, 1, 0, 150, 0))
    print(load_keypoint_files(2, 1, 1, 150, 0))
    print(load_keypoint_files(2, 1, 2, 150, 0))
    # main()
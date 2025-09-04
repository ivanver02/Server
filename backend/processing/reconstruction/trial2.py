from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Asegurar que config esté en el path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from camera import Camera
from triangulation_svd import triangulate_frame_svd
from triangulation_bundle_adjustment import refine_frame_bundle_adjustment
from reprojection import reprojection_error
from pose_estimation import triangulate_with_pose_estimation
from pose_estimation_rigorous import estimate_extrinsics_rigorous
from config.camera_intrinsics import CAMERA_INTRINSICS
from typing import Tuple, Dict

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


def create_cameras_from_config() -> Dict[str, Camera]:
    """Crea las cámaras usando la configuración de intrínsecos."""
    cameras = {}
    for cam_id in ["camera0", "camera1", "camera2"]:
        # Crear cámara base
        cam = Camera.create(cam_id)
        # Los extrínsecos se estimarán posteriormente
        cameras[cam_id] = cam
    return cameras


def prepare_frame_data() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Prepara los datos del frame con los keypoints y confianzas."""
    return {
        "camera0": (coordinates_camera_0.copy(), confidences_camera_0.copy()),
        "camera1": (coordinates_camera_1.copy(), confidences_camera_1.copy()),
        "camera2": (coordinates_camera_2.copy(), confidences_camera_2.copy()),
    }


def main():
    """Función principal de reconstrucción 3D."""
    print("=== Reconstrucción 3D con datos reales ===")
    print("Iniciando script...")
    
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    print("Semilla aleatoria fijada en 42 para reproducibilidad")
    
    # Configuración
    CONFIDENCE_THRESHOLD = 0.5
    BASELINE_02 = 0.72  # 72 cm entre cámaras 0 y 2
    
    # Preparar datos
    cameras = create_cameras_from_config()
    frame_keypoints = prepare_frame_data()
    
    print(f"Umbral de confianza: {CONFIDENCE_THRESHOLD}")
    print(f"Baseline cámaras 0-2: {BASELINE_02}m")
    
    # Contar puntos válidos inicialmente
    coords_0, conf_0 = frame_keypoints["camera0"]
    coords_1, conf_1 = frame_keypoints["camera1"]
    coords_2, conf_2 = frame_keypoints["camera2"]
    
    valid_mask = (conf_0 > CONFIDENCE_THRESHOLD) & \
                 (conf_1 > CONFIDENCE_THRESHOLD) & \
                 (conf_2 > CONFIDENCE_THRESHOLD)
    
    print(f"Puntos con confianza > {CONFIDENCE_THRESHOLD} en todas las cámaras: {np.sum(valid_mask)}")
    
    # Método Riguroso: Estimación con geometría epipolar
    print("\n=== Estimación Rigurosa con Geometría Epipolar ===")
    try:
        cameras_rigorous = estimate_extrinsics_rigorous(
            cameras, frame_keypoints, CONFIDENCE_THRESHOLD, BASELINE_02
        )
        
        # Mostrar resultados de calibración rigurosa
        print("\n--- Parámetros Extrínsecos Estimados ---")
        for cam_id, cam in cameras_rigorous.items():
            if cam_id != "camera0":
                print(f"\n{cam_id}:")
                print(f"  R = \n{cam.R}")
                print(f"  t = {cam.t.flatten()}")
                print(f"  Baseline = {np.linalg.norm(cam.t):.3f}m")
        
        # === PARTE 1: Triangulación SVD ===
        print("\n" + "="*50)
        print("PARTE 1: TRIANGULACIÓN SVD (Sin refinamiento)")
        print("="*50)
        
        points_3d_svd = triangulate_frame_svd(
            cameras_rigorous, frame_keypoints, confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        svd_count = np.sum(~np.isnan(points_3d_svd[:, 0]))
        print(f"Triangulación SVD: {svd_count}/{len(points_3d_svd)} puntos válidos")
        
        # Errores de reproyección con SVD
        errors_svd = reprojection_error(points_3d_svd, cameras_rigorous, frame_keypoints)
        print(f"\nErrores de reproyección (SVD):")
        for cam_id, error in errors_svd.items():
            print(f"  {cam_id}: {error:.2f} píxeles")
        
        # Estadísticas SVD
        if svd_count > 0:
            valid_svd = points_3d_svd[~np.isnan(points_3d_svd[:, 0])]
            print(f"\nEstadísticas puntos 3D (SVD):")
            print(f"  X: min={np.min(valid_svd[:, 0]):.3f}, max={np.max(valid_svd[:, 0]):.3f}")
            print(f"  Y: min={np.min(valid_svd[:, 1]):.3f}, max={np.max(valid_svd[:, 1]):.3f}")
            print(f"  Z: min={np.min(valid_svd[:, 2]):.3f}, max={np.max(valid_svd[:, 2]):.3f}")
        
        # === PARTE 2: Bundle Adjustment ===
        print("\n" + "="*50)
        print("PARTE 2: BUNDLE ADJUSTMENT (Refinamiento)")
        print("="*50)
        
        if svd_count > 0:
            points_3d_ba = refine_frame_bundle_adjustment(
                points_3d_svd, cameras_rigorous, frame_keypoints
            )
            
            ba_count = np.sum(~np.isnan(points_3d_ba[:, 0]))
            print(f"Bundle Adjustment: {svd_count} -> {ba_count} puntos válidos")
            
            # Errores de reproyección con Bundle Adjustment
            errors_ba = reprojection_error(points_3d_ba, cameras_rigorous, frame_keypoints)
            print(f"\nErrores de reproyección (Bundle Adjustment):")
            for cam_id, error in errors_ba.items():
                print(f"  {cam_id}: {error:.2f} píxeles")
            
            # Estadísticas Bundle Adjustment
            if ba_count > 0:
                valid_ba = points_3d_ba[~np.isnan(points_3d_ba[:, 0])]
                print(f"\nEstadísticas puntos 3D (Bundle Adjustment):")
                print(f"  X: min={np.min(valid_ba[:, 0]):.3f}, max={np.max(valid_ba[:, 0]):.3f}")
                print(f"  Y: min={np.min(valid_ba[:, 1]):.3f}, max={np.max(valid_ba[:, 1]):.3f}")
                print(f"  Z: min={np.min(valid_ba[:, 2]):.3f}, max={np.max(valid_ba[:, 2]):.3f}")
            
            # === COMPARACIÓN ===
            print("\n" + "="*50)
            print("COMPARACIÓN SVD vs BUNDLE ADJUSTMENT")
            print("="*50)
            
            print("Mejora en errores de reproyección:")
            for cam_id in errors_svd.keys():
                if cam_id in errors_ba:
                    improvement = errors_svd[cam_id] - errors_ba[cam_id]
                    print(f"  {cam_id}: {errors_svd[cam_id]:.2f} -> {errors_ba[cam_id]:.2f} px ({improvement:+.2f} px)")
            
            avg_error_svd = np.mean(list(errors_svd.values()))
            avg_error_ba = np.mean(list(errors_ba.values()))
            total_improvement = avg_error_svd - avg_error_ba
            print(f"\nError promedio: {avg_error_svd:.2f} -> {avg_error_ba:.2f} px ({total_improvement:+.2f} px)")
        
        else:
            print("No hay puntos válidos para Bundle Adjustment")
        
    except Exception as e:
        print(f"Error en método riguroso: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

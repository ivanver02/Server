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
from full_bundle_adjustment import full_bundle_adjustment, print_camera_changes
from reprojection import reprojection_error
from pose_estimation import triangulate_with_pose_estimation
from pose_estimation_rigorous import estimate_extrinsics_rigorous
from config.camera_intrinsics import CAMERA_INTRINSICS
from typing import Tuple, Dict
import traceback

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


def calculate_distance_from_scaled_points(points_3d: np.ndarray, cameras: Dict[str, Camera], scale_factor: float):
    """Calcula la distancia a las cámaras usando puntos escalados."""
    
    # Aplicar escala a los puntos
    scaled_points = points_3d * scale_factor
    
    # Calcular distancia promedio de todos los puntos válidos a todas las cámaras
    all_distances = []
    
    for cam_id, cam in cameras.items():
        # Posición de la cámara (considerando que camera0 está en origen)
        if cam_id == "camera0":
            cam_pos = np.array([0.0, 0.0, 0.0])
        else:
            # Para otras cámaras, la posición es -t (transformación inversa)
            cam_pos = -cam.t.flatten() * scale_factor  # También escalar posiciones de cámara
        
        # Calcular distancias de puntos válidos a esta cámara
        for point in scaled_points:
            if not np.isnan(point[0]):  # Solo puntos válidos
                distance = np.linalg.norm(point - cam_pos)
                all_distances.append(distance)
    
    # Distancia promedio estimada
    estimated_avg_distance = np.mean(all_distances)
    
    print(f"\n=== DISTANCIA ESTIMADA A LAS CÁMARAS (ESCALADA) ===")
    print(f"Basado en antebrazo de 30 cm como referencia")
    print(f"Distancia promedio estimada: {estimated_avg_distance:.2f} metros")
    
    return estimated_avg_distance


def calculate_scale_factor_from_forearm(points_3d: np.ndarray, real_forearm_length_cm: float = 30.0):
    """Calcula el factor de escala basado en la longitud real del antebrazo (codo a muñeca)."""
    
    # Índices para codo y muñeca (derecha e izquierda)
    # 7: Codo_izq, 8: Codo_der, 9: Muñeca_izq, 10: Muñeca_der
    forearm_measurements = []
    
    # Antebrazo izquierdo (codo_izq a muñeca_izq)
    if not np.isnan(points_3d[7, 0]) and not np.isnan(points_3d[9, 0]):
        left_forearm = np.linalg.norm(points_3d[7] - points_3d[9])
        forearm_measurements.append(("Antebrazo izquierdo", left_forearm))
    
    # Antebrazo derecho (codo_der a muñeca_der)
    if not np.isnan(points_3d[8, 0]) and not np.isnan(points_3d[10, 0]):
        right_forearm = np.linalg.norm(points_3d[8] - points_3d[10])
        forearm_measurements.append(("Antebrazo derecho", right_forearm))
    
    if not forearm_measurements:
        print("❌ ERROR: No se pueden medir los antebrazos (puntos no válidos)")
        return 1.0
    
    # Usar el promedio de las medidas disponibles
    avg_forearm_length_m = np.mean([length for _, length in forearm_measurements])
    real_forearm_length_m = real_forearm_length_cm / 100.0  # convertir a metros
    
    # Factor de escala
    scale_factor = real_forearm_length_m / avg_forearm_length_m
    
    print(f"\n=== CÁLCULO DE FACTOR DE ESCALA (BASADO EN ANTEBRAZO) ===")
    print(f"Longitud real del antebrazo: {real_forearm_length_cm:.1f} cm")
    for name, length in forearm_measurements:
        print(f"{name} estimado: {length*100:.1f} cm")
    print(f"Longitud promedio estimada: {avg_forearm_length_m*100:.1f} cm")
    print(f"Factor de escala calculado: {scale_factor:.4f}")
    
    return scale_factor


def analyze_body_measurements_scaled(points_3d: np.ndarray, method_name: str, scale_factor: float):
    """Analiza las medidas corporales con escala corregida basada en longitud del antebrazo."""
    
    # Aplicar factor de escala a todos los puntos
    scaled_points = points_3d * scale_factor
    
    # Definición de keypoints según el esqueleto típico
    keypoint_names = [
        "Nariz", "Ojo_izq", "Ojo_der", "Oreja_izq", "Oreja_der",
        "Hombro_izq", "Hombro_der", "Codo_izq", "Codo_der", 
        "Muñeca_izq", "Muñeca_der", "Cadera_izq", "Cadera_der",
        "Rodilla_izq", "Rodilla_der", "Tobillo_izq", "Tobillo_der",
        "Extra_17", "Extra_18", "Extra_19", "Extra_20", "Extra_21", "Extra_22"
    ]
    
    def distance_3d(p1_idx: int, p2_idx: int) -> float:
        """Calcula distancia euclidiana entre dos puntos 3D escalados."""
        if (p1_idx >= len(scaled_points) or p2_idx >= len(scaled_points) or 
            np.isnan(scaled_points[p1_idx, 0]) or np.isnan(scaled_points[p2_idx, 0])):
            return np.nan
        return np.linalg.norm(scaled_points[p1_idx] - scaled_points[p2_idx])
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE MEDIDAS CORPORALES ESCALADAS ({method_name})")
    print(f"Referencia: Antebrazo = 30.0 cm, Factor de escala: {scale_factor:.4f}")
    print(f"{'='*70}")
    
    # Medidas principales del cuerpo
    measurements = []
    
    # Cabeza y cuello
    measurements.extend([
        ("Ancho cara (ojo_izq - ojo_der)", distance_3d(1, 2), "7-10 cm"),
        ("Distancia ojos-nariz promedio", 
         np.nanmean([distance_3d(0, 1), distance_3d(0, 2)]), "2-4 cm"),
    ])
    
    # Torso
    measurements.extend([
        ("Ancho hombros", distance_3d(5, 6), "35-45 cm"),
        ("Alto torso (hombro_izq - cadera_izq)", distance_3d(5, 11), "50-70 cm"),
        ("Alto torso (hombro_der - cadera_der)", distance_3d(6, 12), "50-70 cm"),
        ("Ancho caderas", distance_3d(11, 12), "25-35 cm"),
    ])
    
    # Brazo izquierdo
    measurements.extend([
        ("Brazo izq (hombro-codo)", distance_3d(5, 7), "28-36 cm"),
        ("Antebrazo izq (codo-muñeca)", distance_3d(7, 9), "23-30 cm"),
        ("Brazo completo izq (hombro-muñeca)", distance_3d(5, 9), "55-70 cm"),
    ])
    
    # Brazo derecho
    measurements.extend([
        ("Brazo der (hombro-codo)", distance_3d(6, 8), "28-36 cm"),
        ("Antebrazo der (codo-muñeca)", distance_3d(8, 10), "23-30 cm"),
        ("Brazo completo der (hombro-muñeca)", distance_3d(6, 10), "55-70 cm"),
    ])
    
    # Pierna izquierda
    measurements.extend([
        ("Muslo izq (cadera-rodilla)", distance_3d(11, 13), "35-50 cm"),
        ("Pantorrilla izq (rodilla-tobillo)", distance_3d(13, 15), "35-45 cm"),
        ("Pierna completa izq (cadera-tobillo)", distance_3d(11, 15), "75-100 cm"),
    ])
    
    # Pierna derecha
    measurements.extend([
        ("Muslo der (cadera-rodilla)", distance_3d(12, 14), "35-50 cm"),
        ("Pantorrilla der (rodilla-tobillo)", distance_3d(14, 16), "35-45 cm"),
        ("Pierna completa der (cadera-tobillo)", distance_3d(12, 16), "75-100 cm"),
    ])
    
    # Medidas adicionales
    measurements.extend([
        ("Estatura aprox (cabeza-tobillo_izq)", 
         distance_3d(0, 15) if not np.isnan(distance_3d(0, 15)) else np.nan, "150-190 cm"),
        ("Estatura aprox (cabeza-tobillo_der)", 
         distance_3d(0, 16) if not np.isnan(distance_3d(0, 16)) else np.nan, "150-190 cm"),
        ("Envergadura (muñeca_izq - muñeca_der)", distance_3d(9, 10), "150-180 cm"),
    ])
    
    # Mostrar resultados
    print(f"{'Medida':<40} | {'Valor':<12} | {'Rango Normal':<15} | {'Estado'}")
    print("-" * 85)
    
    valid_measurements = 0
    realistic_measurements = 0
    
    for name, value, normal_range in measurements:
        if np.isnan(value):
            status = "N/A"
            value_str = "N/A"
        else:
            value_cm = value * 100  # convertir a cm
            value_str = f"{value_cm:.1f} cm"
            
            # Análisis básico de realismo (rangos aproximados)
            if "cara" in name.lower() or "ojos" in name.lower():
                realistic = 3 <= value_cm <= 15
            elif "hombros" in name.lower():
                realistic = 25 <= value_cm <= 55
            elif "torso" in name.lower():
                realistic = 40 <= value_cm <= 80
            elif "caderas" in name.lower():
                realistic = 20 <= value_cm <= 45
            elif "brazo" in name.lower() and "completo" not in name.lower():
                realistic = 20 <= value_cm <= 45
            elif "brazo completo" in name.lower():
                realistic = 45 <= value_cm <= 80
            elif "muslo" in name.lower():
                realistic = 30 <= value_cm <= 60
            elif "pantorrilla" in name.lower():
                realistic = 30 <= value_cm <= 55
            elif "pierna completa" in name.lower():
                realistic = 65 <= value_cm <= 110
            elif "estatura" in name.lower():
                realistic = 140 <= value_cm <= 200
            elif "envergadura" in name.lower():
                realistic = 140 <= value_cm <= 200
            else:
                realistic = True  # Por defecto aceptar
                
            status = "✅ OK" if realistic else "⚠️ Fuera rango"
            if realistic:
                realistic_measurements += 1
            valid_measurements += 1
        
        print(f"{name:<40} | {value_str:<12} | {normal_range:<15} | {status}")
    
    # Resumen
    print(f"\n{'='*70}")
    print(f"RESUMEN DE VALIDACIÓN ESCALADA ({method_name})")
    print(f"{'='*70}")
    print(f"Medidas válidas: {valid_measurements}/{len(measurements)}")
    print(f"Medidas realistas: {realistic_measurements}/{valid_measurements} ({realistic_measurements/max(valid_measurements,1)*100:.1f}%)")
    
    if realistic_measurements / max(valid_measurements, 1) > 0.8:
        print("🟢 EXCELENTE: Las proporciones corporales escaladas son muy realistas")
    elif realistic_measurements / max(valid_measurements, 1) > 0.6:
        print("🟡 BUENO: Las proporciones corporales escaladas son aceptables")
    else:
        print("🔴 PROBLEMA: Las proporciones corporales escaladas siguen siendo irrealistas")


def analyze_body_measurements(points_3d: np.ndarray, method_name: str):
    """Analiza las medidas corporales entre keypoints anatómicos."""
    
    # Definición de keypoints según el esqueleto típico (ajustar según tu modelo)
    keypoint_names = [
        "Nariz", "Ojo_izq", "Ojo_der", "Oreja_izq", "Oreja_der",
        "Hombro_izq", "Hombro_der", "Codo_izq", "Codo_der", 
        "Muñeca_izq", "Muñeca_der", "Cadera_izq", "Cadera_der",
        "Rodilla_izq", "Rodilla_der", "Tobillo_izq", "Tobillo_der",
        "Extra_17", "Extra_18", "Extra_19", "Extra_20", "Extra_21", "Extra_22"
    ]
    
    def distance_3d(p1_idx: int, p2_idx: int) -> float:
        """Calcula distancia euclidiana entre dos puntos 3D."""
        if (p1_idx >= len(points_3d) or p2_idx >= len(points_3d) or 
            np.isnan(points_3d[p1_idx, 0]) or np.isnan(points_3d[p2_idx, 0])):
            return np.nan
        return np.linalg.norm(points_3d[p1_idx] - points_3d[p2_idx])
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE MEDIDAS CORPORALES ({method_name})")
    print(f"{'='*60}")
    
    # Medidas principales del cuerpo
    measurements = []
    
    # Cabeza y cuello
    measurements.extend([
        ("Ancho cara (ojo_izq - ojo_der)", distance_3d(1, 2), "7-10 cm"),
        ("Distancia ojos-nariz promedio", 
         np.nanmean([distance_3d(0, 1), distance_3d(0, 2)]), "2-4 cm"),
    ])
    
    # Torso
    measurements.extend([
        ("Ancho hombros", distance_3d(5, 6), "35-45 cm"),
        ("Alto torso (hombro_izq - cadera_izq)", distance_3d(5, 11), "50-70 cm"),
        ("Alto torso (hombro_der - cadera_der)", distance_3d(6, 12), "50-70 cm"),
        ("Ancho caderas", distance_3d(11, 12), "25-35 cm"),
    ])
    
    # Brazo izquierdo
    measurements.extend([
        ("Brazo izq (hombro-codo)", distance_3d(5, 7), "28-36 cm"),
        ("Antebrazo izq (codo-muñeca)", distance_3d(7, 9), "23-30 cm"),
        ("Brazo completo izq (hombro-muñeca)", distance_3d(5, 9), "55-70 cm"),
    ])
    
    # Brazo derecho
    measurements.extend([
        ("Brazo der (hombro-codo)", distance_3d(6, 8), "28-36 cm"),
        ("Antebrazo der (codo-muñeca)", distance_3d(8, 10), "23-30 cm"),
        ("Brazo completo der (hombro-muñeca)", distance_3d(6, 10), "55-70 cm"),
    ])
    
    # Pierna izquierda
    measurements.extend([
        ("Muslo izq (cadera-rodilla)", distance_3d(11, 13), "35-50 cm"),
        ("Pantorrilla izq (rodilla-tobillo)", distance_3d(13, 15), "35-45 cm"),
        ("Pierna completa izq (cadera-tobillo)", distance_3d(11, 15), "75-100 cm"),
    ])
    
    # Pierna derecha
    measurements.extend([
        ("Muslo der (cadera-rodilla)", distance_3d(12, 14), "35-50 cm"),
        ("Pantorrilla der (rodilla-tobillo)", distance_3d(14, 16), "35-45 cm"),
        ("Pierna completa der (cadera-tobillo)", distance_3d(12, 16), "75-100 cm"),
    ])
    
    # Medidas adicionales
    measurements.extend([
        ("Estatura aprox (cabeza-tobillo_izq)", 
         distance_3d(0, 15) if not np.isnan(distance_3d(0, 15)) else np.nan, "150-190 cm"),
        ("Estatura aprox (cabeza-tobillo_der)", 
         distance_3d(0, 16) if not np.isnan(distance_3d(0, 16)) else np.nan, "150-190 cm"),
        ("Envergadura (muñeca_izq - muñeca_der)", distance_3d(9, 10), "150-180 cm"),
    ])
    
    # Mostrar resultados
    print(f"{'Medida':<40} | {'Valor':<12} | {'Rango Normal':<15} | {'Estado'}")
    print("-" * 85)
    
    valid_measurements = 0
    realistic_measurements = 0
    
    for name, value, normal_range in measurements:
        if np.isnan(value):
            status = "❌ N/A"
            value_str = "N/A"
        else:
            value_cm = value * 100  # convertir a cm
            value_str = f"{value_cm:.1f} cm"
            
            # Análisis básico de realismo (rangos aproximados)
            if "cara" in name.lower() or "ojos" in name.lower():
                realistic = 3 <= value_cm <= 15
            elif "hombros" in name.lower():
                realistic = 25 <= value_cm <= 55
            elif "torso" in name.lower():
                realistic = 40 <= value_cm <= 80
            elif "caderas" in name.lower():
                realistic = 20 <= value_cm <= 45
            elif "brazo" in name.lower() and "completo" not in name.lower():
                realistic = 20 <= value_cm <= 45
            elif "brazo completo" in name.lower():
                realistic = 45 <= value_cm <= 80
            elif "muslo" in name.lower():
                realistic = 30 <= value_cm <= 60
            elif "pantorrilla" in name.lower():
                realistic = 30 <= value_cm <= 55
            elif "pierna completa" in name.lower():
                realistic = 65 <= value_cm <= 110
            elif "estatura" in name.lower():
                realistic = 140 <= value_cm <= 200
            elif "envergadura" in name.lower():
                realistic = 140 <= value_cm <= 200
            else:
                realistic = True  # Por defecto aceptar
                
            status = "✅ OK" if realistic else "⚠️ Fuera rango"
            if realistic:
                realistic_measurements += 1
            valid_measurements += 1
        
        print(f"{name:<40} | {value_str:<12} | {normal_range:<15} | {status}")
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"RESUMEN DE VALIDACIÓN ({method_name})")
    print(f"{'='*60}")
    print(f"Medidas válidas: {valid_measurements}/{len(measurements)}")
    print(f"Medidas realistas: {realistic_measurements}/{valid_measurements} ({realistic_measurements/max(valid_measurements,1)*100:.1f}%)")
    
    if realistic_measurements / max(valid_measurements, 1) > 0.8:
        print("🟢 EXCELENTE: Las proporciones corporales son muy realistas")
    elif realistic_measurements / max(valid_measurements, 1) > 0.6:
        print("🟡 BUENO: Las proporciones corporales son aceptables")
    else:
        print("🔴 PROBLEMA: Las proporciones corporales parecen irrealistas")


def print_3d_coordinates_scaled(points_3d: np.ndarray, method_name: str, scale_factor: float):
    """Imprime las coordenadas 3D escaladas basadas en la longitud del antebrazo."""
    
    # Aplicar factor de escala
    scaled_points = points_3d * scale_factor
    
    keypoint_names = [
        "Nariz", "Ojo_izq", "Ojo_der", "Oreja_izq", "Oreja_der",
        "Hombro_izq", "Hombro_der", "Codo_izq", "Codo_der", 
        "Muñeca_izq", "Muñeca_der", "Cadera_izq", "Cadera_der",
        "Rodilla_izq", "Rodilla_der", "Tobillo_izq", "Tobillo_der",
        "Extra_17", "Extra_18", "Extra_19", "Extra_20", "Extra_21", "Extra_22"
    ]
    
    print(f"\n{'='*80}")
    print(f"COORDENADAS 3D ESCALADAS DE KEYPOINTS ({method_name})")
    print(f"Referencia: Antebrazo = 30.0 cm, Factor de escala: {scale_factor:.4f}")
    print(f"{'='*80}")
    
    print(f"{'Keypoint':<15} | {'X (m)':<10} | {'Y (m)':<10} | {'Z (m)':<10} | {'Estado'}")
    print("-" * 80)
    
    valid_count = 0
    for i, (name, point) in enumerate(zip(keypoint_names, scaled_points)):
        if np.isnan(point[0]):
            status = "❌ No válido"
            x_str = y_str = z_str = "N/A"
        else:
            status = "✅ Válido"
            x_str = f"{point[0]:.3f}"
            y_str = f"{point[1]:.3f}"
            z_str = f"{point[2]:.3f}"
            valid_count += 1
        
        print(f"{name:<15} | {x_str:<10} | {y_str:<10} | {z_str:<10} | {status}")
    
    print(f"\nPuntos 3D válidos (escalados): {valid_count}/{len(scaled_points)}")


def print_3d_coordinates(points_3d: np.ndarray, method_name: str):
    """Imprime las coordenadas 3D de cada keypoint."""
    
    keypoint_names = [
        "Nariz", "Ojo_izq", "Ojo_der", "Oreja_izq", "Oreja_der",
        "Hombro_izq", "Hombro_der", "Codo_izq", "Codo_der", 
        "Muñeca_izq", "Muñeca_der", "Cadera_izq", "Cadera_der",
        "Rodilla_izq", "Rodilla_der", "Tobillo_izq", "Tobillo_der",
        "Extra_17", "Extra_18", "Extra_19", "Extra_20", "Extra_21", "Extra_22"
    ]
    
    print(f"\n{'='*70}")
    print(f"COORDENADAS 3D DE KEYPOINTS ({method_name})")
    print(f"{'='*70}")
    
    print(f"{'Keypoint':<15} | {'X (m)':<10} | {'Y (m)':<10} | {'Z (m)':<10} | {'Estado'}")
    print("-" * 70)
    
    valid_count = 0
    for i, (name, point) in enumerate(zip(keypoint_names, points_3d)):
        if np.isnan(point[0]):
            status = "❌ No válido"
            x_str = y_str = z_str = "N/A"
        else:
            status = "✅ Válido"
            x_str = f"{point[0]:.3f}"
            y_str = f"{point[1]:.3f}"
            z_str = f"{point[2]:.3f}"
            valid_count += 1
        
        print(f"{name:<15} | {x_str:<10} | {y_str:<10} | {z_str:<10} | {status}")
    
    print(f"\nPuntos 3D válidos: {valid_count}/{len(points_3d)}")


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
        
        # Mostrar coordenadas 3D para SVD
        print_3d_coordinates(points_3d_svd, "SVD")
        
        # Calcular factor de escala basado en longitud real del antebrazo (30 cm)
        scale_factor_svd = calculate_scale_factor_from_forearm(points_3d_svd, real_forearm_length_cm=30.0)
        
        # Mostrar coordenadas 3D escaladas para SVD
        print_3d_coordinates_scaled(points_3d_svd, "SVD", scale_factor_svd)
        
        # Análisis de medidas corporales escaladas para SVD
        analyze_body_measurements_scaled(points_3d_svd, "SVD", scale_factor_svd)
        
        # === PARTE 2: Bundle Adjustment Simple ===
        print("\n" + "="*50)
        print("PARTE 2: BUNDLE ADJUSTMENT SIMPLE (Solo puntos 3D)")
        print("="*50)
        
        if svd_count > 0:
            points_3d_ba = refine_frame_bundle_adjustment(
                points_3d_svd, cameras_rigorous, frame_keypoints
            )
            
            ba_count = np.sum(~np.isnan(points_3d_ba[:, 0]))
            print(f"Bundle Adjustment Simple: {svd_count} -> {ba_count} puntos válidos")
            
            # Errores de reproyección con Bundle Adjustment Simple
            errors_ba = reprojection_error(points_3d_ba, cameras_rigorous, frame_keypoints)
            print(f"\nErrores de reproyección (Bundle Adjustment Simple):")
            for cam_id, error in errors_ba.items():
                print(f"  {cam_id}: {error:.2f} píxeles")
            
            # Mostrar coordenadas 3D para Bundle Adjustment Simple
            print_3d_coordinates(points_3d_ba, "Bundle Adjustment Simple")
            
            # Calcular factor de escala basado en longitud real del antebrazo (30 cm)
            scale_factor = calculate_scale_factor_from_forearm(points_3d_ba, real_forearm_length_cm=30.0)
            
            # Mostrar coordenadas 3D escaladas para Bundle Adjustment Simple
            print_3d_coordinates_scaled(points_3d_ba, "Bundle Adjustment Simple", scale_factor)
            
            # Análisis de medidas corporales escaladas para Bundle Adjustment Simple
            analyze_body_measurements_scaled(points_3d_ba, "Bundle Adjustment Simple", scale_factor)
            
        else:
            print("❌ ERROR: No hay puntos válidos de SVD para Bundle Adjustment Simple")
            points_3d_ba = points_3d_svd
            errors_ba = errors_svd
        
        # === PARTE 3: Bundle Adjustment Completo ===
        print("\n" + "="*50)
        print("PARTE 3: BUNDLE ADJUSTMENT COMPLETO (Puntos 3D + Extrínsecos)")
        print("="*50)
        
        if svd_count > 0:
            points_3d_full_ba, cameras_optimized = full_bundle_adjustment(
                points_3d_svd, cameras_rigorous, frame_keypoints
            )
            
            full_ba_count = np.sum(~np.isnan(points_3d_full_ba[:, 0]))
            print(f"Bundle Adjustment Completo: {svd_count} -> {full_ba_count} puntos válidos")
            
            # Mostrar cambios en parámetros de cámaras
            print_camera_changes(cameras_rigorous, cameras_optimized)
            
            # Errores de reproyección con Bundle Adjustment Completo
            errors_full_ba = reprojection_error(points_3d_full_ba, cameras_optimized, frame_keypoints)
            print(f"\nErrores de reproyección (Bundle Adjustment Completo):")
            for cam_id, error in errors_full_ba.items():
                print(f"  {cam_id}: {error:.2f} píxeles")
            
            # Mostrar coordenadas 3D para Bundle Adjustment Completo
            print_3d_coordinates(points_3d_full_ba, "Bundle Adjustment Completo")
            
            # Calcular factor de escala basado en longitud real del antebrazo (30 cm)
            scale_factor_full = calculate_scale_factor_from_forearm(points_3d_full_ba, real_forearm_length_cm=30.0)
            
            # Mostrar coordenadas 3D escaladas para Bundle Adjustment Completo
            print_3d_coordinates_scaled(points_3d_full_ba, "Bundle Adjustment Completo", scale_factor_full)
            
            # Análisis de medidas corporales escaladas para Bundle Adjustment Completo
            analyze_body_measurements_scaled(points_3d_full_ba, "Bundle Adjustment Completo", scale_factor_full)
            
            # Calcular distancia estimada a las cámaras con escalado
            estimated_distance = calculate_distance_from_scaled_points(points_3d_full_ba, cameras_optimized, scale_factor_full)
            
            # También mostrar análisis sin escalar para comparación
            print("\n" + "="*50)
            print("ANÁLISIS SIN ESCALAR (para comparación)")
            print("="*50)
            analyze_body_measurements(points_3d_full_ba, "Bundle Adjustment Completo - Sin Escalar")
            
        else:
            print("❌ ERROR: No hay puntos válidos de SVD para Bundle Adjustment Completo")
            points_3d_full_ba = points_3d_svd
            cameras_optimized = cameras_rigorous
            errors_full_ba = errors_svd
        
        # === COMPARACIÓN COMPLETA ===
        print("\n" + "="*60)
        print("COMPARACIÓN COMPLETA DE MÉTODOS")
        print("="*60)
        
        print("\n--- Errores de reproyección ---")
        methods_errors = [
            ("SVD", errors_svd),
            ("Bundle Adj. Simple", errors_ba),
            ("Bundle Adj. Completo", errors_full_ba)
        ]
        
        print(f"{'Método':<20} | {'Camera0':<10} | {'Camera1':<10} | {'Camera2':<10} | {'Promedio':<10}")
        print("-" * 75)
        
        for method_name, errors in methods_errors:
            avg_error = np.mean(list(errors.values()))
            print(f"{method_name:<20} | {errors['camera0']:<10.2f} | {errors['camera1']:<10.2f} | {errors['camera2']:<10.2f} | {avg_error:<10.2f}")
        
        print("\n--- Mejoras en errores ---")
        for cam_id in errors_svd.keys():
            improvement_simple = errors_svd[cam_id] - errors_ba[cam_id]
            improvement_full = errors_svd[cam_id] - errors_full_ba[cam_id]
            print(f"{cam_id}: Simple {improvement_simple:+.2f} px, Completo {improvement_full:+.2f} px")
        
        avg_error_svd = np.mean(list(errors_svd.values()))
        avg_error_ba = np.mean(list(errors_ba.values()))
        avg_error_full_ba = np.mean(list(errors_full_ba.values()))
        
        improvement_simple = avg_error_svd - avg_error_ba
        improvement_full = avg_error_svd - avg_error_full_ba
        
        print(f"\nPromedio: Simple {improvement_simple:+.2f} px, Completo {improvement_full:+.2f} px")
        
        # === RESUMEN FINAL ===
        print("\n" + "="*60)
        print("RESUMEN FINAL ESCALADO")
        print("="*60)
        
        print("Bundle Adjustment Completo ha optimizado tanto los puntos 3D como")
        print("los parámetros extrínsecos de las cámaras para minimizar el error total.")
        print(f"Factor de escala final: {scale_factor_full:.4f}")
        print(f"Distancia estimada final: {estimated_distance:.2f} metros")
        
    except Exception as e:
        print(f"Error en método riguroso: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

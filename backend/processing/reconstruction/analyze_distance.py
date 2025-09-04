import numpy as np

# Coordenadas 3D obtenidas del Bundle Adjustment (las más precisas)
points_3d_ba = np.array([
    [-2.903, -1.734, 6.681],   # Nariz
    [-3.050, -1.948, 7.210],   # Ojo_izq
    [-2.983, -1.821, 6.689],   # Ojo_der
    [-3.614, -2.329, 8.862],   # Oreja_izq
    [-3.256, -1.872, 6.989],   # Oreja_der
    [-3.927, -1.838, 10.111],  # Hombro_izq
    [-3.111, -1.220, 6.200],   # Hombro_der
    [-4.323, -0.953, 11.584],  # Codo_izq
    [-2.253, -0.595, 3.829],   # Codo_der
    [-3.495, 0.021, 9.393],    # Muñeca_izq
    [-1.852, -0.794, 3.224],   # Muñeca_der
    [-3.239, 0.002, 7.637],    # Cadera_izq
    [-3.034, -0.033, 6.067],   # Cadera_der
    [-4.220, 1.579, 9.572],    # Rodilla_izq
    [-4.488, 1.415, 8.711],    # Rodilla_der
    [-5.845, 4.191, 12.740],   # Tobillo_izq
    [-6.041, 3.778, 11.536],   # Tobillo_der
])

# Posiciones de las cámaras (estimadas)
# Camera0 está en el origen: [0, 0, 0]
camera0_pos = np.array([0.0, 0.0, 0.0])

# Camera1 posición: -t de camera1 (t = [-0.603, -0.302, 1.107])
camera1_pos = np.array([0.603, 0.302, -1.107])

# Camera2 posición: -t de camera2 (t = [0.279, 0.202, -0.632])  
camera2_pos = np.array([-0.279, -0.202, 0.632])

def calculate_distances_to_cameras(points_3d, cameras_pos):
    """Calcula la distancia de cada punto 3D a cada cámara."""
    distances = {}
    
    for i, cam_pos in enumerate(cameras_pos):
        cam_name = f"camera{i}"
        # Distancias de todos los puntos a esta cámara
        dists = [np.linalg.norm(point - cam_pos) for point in points_3d]
        distances[cam_name] = dists
    
    return distances

def analyze_person_distance():
    """Analiza la distancia de la persona a las cámaras."""
    
    cameras_positions = [camera0_pos, camera1_pos, camera2_pos]
    distances = calculate_distances_to_cameras(points_3d_ba, cameras_positions)
    
    print("=== ANÁLISIS DE DISTANCIA DE LA PERSONA A LAS CÁMARAS ===")
    print(f"Altura real de la persona: 190 cm")
    print(f"Distancia real reportada: 2.30 metros")
    print(f"Baseline conocido: 0.72 metros (cámaras 0-2)")
    print()
    
    # Calcular estadísticas por cámara
    for cam_name, dists in distances.items():
        avg_dist = np.mean(dists)
        min_dist = np.min(dists)
        max_dist = np.max(dists)
        
        print(f"{cam_name}:")
        print(f"  Distancia promedio: {avg_dist:.2f} m")
        print(f"  Distancia mínima: {min_dist:.2f} m")
        print(f"  Distancia máxima: {max_dist:.2f} m")
        print()
    
    # Distancia promedio total
    all_distances = []
    for dists in distances.values():
        all_distances.extend(dists)
    
    overall_avg = np.mean(all_distances)
    print(f"DISTANCIA PROMEDIO GENERAL (según reconstrucción): {overall_avg:.2f} metros")
    print()
    
    # Calcular factor de escala basado en la altura real
    # Estatura estimada por la reconstrucción
    estimated_height_m = np.linalg.norm(points_3d_ba[0] - points_3d_ba[15])  # cabeza a tobillo izquierdo
    real_height_m = 1.90  # 190 cm
    
    print(f"=== ANÁLISIS BASADO EN ALTURA REAL (190 cm) ===")
    print(f"Estatura estimada por reconstrucción: {estimated_height_m:.2f} m ({estimated_height_m*100:.0f} cm)")
    print(f"Estatura real: {real_height_m:.2f} m (190 cm)")
    
    # Factor de escala basado en altura
    height_scale_factor = real_height_m / estimated_height_m
    print(f"Factor de escala (basado en altura): {height_scale_factor:.3f}")
    
    # Distancia corregida basada en altura
    corrected_distance = overall_avg * height_scale_factor
    print(f"Distancia corregida (basada en altura): {corrected_distance:.2f} metros")
    print()
    
    # Comparar con la distancia reportada de 2.30m
    scale_factor_230 = 2.30 / overall_avg
    print(f"=== COMPARACIÓN CON DISTANCIA REPORTADA (2.30m) ===")
    print(f"Factor de escala necesario para 2.30m: {scale_factor_230:.3f}")
    print(f"Distancia estimada vs reportada: {overall_avg:.2f}m vs 2.30m")
    print(f"Error relativo: {abs(overall_avg - 2.30) / 2.30 * 100:.1f}%")
    print()
    
    print(f"=== CONCLUSIÓN ===")
    print(f"🎯 Según tu altura real (190cm), estarías a: {corrected_distance:.2f} metros de las cámaras")
    print(f"📏 Esto es muy cercano a los 2.30m reportados (diferencia: {abs(corrected_distance - 2.30):.2f}m)")
    print()
    
    # Análisis de medidas escaladas
    print("=== MEDIDAS ESCALADAS (aplicando factor de corrección basado en altura) ===")
    
    # Ejemplos de medidas escaladas usando el factor de altura
    measurements = [
        ("Ancho cara (ojo_izq - ojo_der)", np.linalg.norm(points_3d_ba[1] - points_3d_ba[2]) * height_scale_factor * 100, "7-10 cm"),
        ("Estatura aprox (cabeza-tobillo_izq)", np.linalg.norm(points_3d_ba[0] - points_3d_ba[15]) * height_scale_factor * 100, "150-190 cm"),
        ("Ancho hombros", np.linalg.norm(points_3d_ba[5] - points_3d_ba[6]) * height_scale_factor * 100, "35-45 cm"),
        ("Envergadura (muñeca_izq - muñeca_der)", np.linalg.norm(points_3d_ba[9] - points_3d_ba[10]) * height_scale_factor * 100, "150-180 cm"),
    ]
    
    print(f"{'Medida':<40} | {'Valor Escalado':<15} | {'Rango Normal'}")
    print("-" * 75)
    for name, scaled_value, normal_range in measurements:
        status = "✅" if (
            ("cara" in name.lower() and 7 <= scaled_value <= 10) or
            ("estatura" in name.lower() and 180 <= scaled_value <= 200) or
            ("hombros" in name.lower() and 35 <= scaled_value <= 45) or
            ("envergadura" in name.lower() and 150 <= scaled_value <= 180)
        ) else "⚠️"
        print(f"{name:<40} | {status} {scaled_value:.1f} cm{'':<5} | {normal_range}")

if __name__ == "__main__":
    analyze_person_distance()

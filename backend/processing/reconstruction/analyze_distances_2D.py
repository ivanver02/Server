import numpy as np
from typing import Dict, List, Tuple

# Datos de keypoints 2D de las tres cámaras (copiados de trial2.py)
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


def filter_valid_keypoints(confidence_threshold: float = 0.5) -> np.ndarray:
    """Filtra keypoints que tienen confianza > threshold en todas las cámaras."""
    
    # Crear máscara de puntos válidos
    valid_mask = (confidences_camera_0 > confidence_threshold) & \
                 (confidences_camera_1 > confidence_threshold) & \
                 (confidences_camera_2 > confidence_threshold)
    
    print(f"=== FILTRADO DE KEYPOINTS 2D ===")
    print(f"Umbral de confianza: {confidence_threshold}")
    print(f"Puntos válidos en todas las cámaras: {np.sum(valid_mask)}/{len(valid_mask)}")
    
    return valid_mask


def calculate_2d_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calcula distancia euclidiana entre dos puntos 2D."""
    return np.linalg.norm(point1 - point2)


def calculate_scale_factor_from_2d_forearm(valid_mask: np.ndarray, real_forearm_length_cm: float = 30.0) -> Dict[str, float]:
    """Calcula factor de escala para cada cámara basado en la longitud del antebrazo en 2D."""
    
    # Índices: 7=Codo_izq, 8=Codo_der, 9=Muñeca_izq, 10=Muñeca_der
    scale_factors = {}
    
    cameras_data = {
        "camera0": coordinates_camera_0,
        "camera1": coordinates_camera_1, 
        "camera2": coordinates_camera_2
    }
    
    print(f"\n=== CÁLCULO DE FACTORES DE ESCALA 2D (ANTEBRAZO = {real_forearm_length_cm} cm) ===")
    
    for cam_name, coords in cameras_data.items():
        forearm_measurements = []
        
        # Antebrazo izquierdo (codo_izq a muñeca_izq)
        if valid_mask[7] and valid_mask[9]:  # Codo_izq y Muñeca_izq válidos
            left_forearm_2d = calculate_2d_distance(coords[7], coords[9])
            forearm_measurements.append(("Izquierdo", left_forearm_2d))
        
        # Antebrazo derecho (codo_der a muñeca_der)
        if valid_mask[8] and valid_mask[10]:  # Codo_der y Muñeca_der válidos
            right_forearm_2d = calculate_2d_distance(coords[8], coords[10])
            forearm_measurements.append(("Derecho", right_forearm_2d))
        
        if forearm_measurements:
            # Usar promedio de las medidas disponibles
            avg_forearm_pixels = np.mean([pixels for _, pixels in forearm_measurements])
            scale_factor = real_forearm_length_cm / avg_forearm_pixels  # cm/pixel
            scale_factors[cam_name] = scale_factor
            
            print(f"\n{cam_name}:")
            for side, pixels in forearm_measurements:
                print(f"  Antebrazo {side}: {pixels:.1f} píxeles")
            print(f"  Promedio: {avg_forearm_pixels:.1f} píxeles")
            print(f"  Factor de escala: {scale_factor:.4f} cm/pixel")
        else:
            scale_factors[cam_name] = None
            print(f"\n{cam_name}: ❌ No se pueden medir antebrazos (puntos no válidos)")
    
    return scale_factors


def analyze_2d_body_measurements(valid_mask: np.ndarray, scale_factors: Dict[str, float]):
    """Analiza medidas corporales 2D escaladas para cada cámara."""
    
    keypoint_names = [
        "Nariz", "Ojo_izq", "Ojo_der", "Oreja_izq", "Oreja_der",
        "Hombro_izq", "Hombro_der", "Codo_izq", "Codo_der", 
        "Muñeca_izq", "Muñeca_der", "Cadera_izq", "Cadera_der",
        "Rodilla_izq", "Rodilla_der", "Tobillo_izq", "Tobillo_der",
        "Extra_17", "Extra_18", "Extra_19", "Extra_20", "Extra_21", "Extra_22"
    ]
    
    cameras_data = {
        "camera0": coordinates_camera_0,
        "camera1": coordinates_camera_1, 
        "camera2": coordinates_camera_2
    }
    
    def distance_2d_scaled(coords: np.ndarray, p1_idx: int, p2_idx: int, scale_factor: float) -> float:
        """Calcula distancia 2D escalada entre dos keypoints."""
        if not (valid_mask[p1_idx] and valid_mask[p2_idx]) or scale_factor is None:
            return np.nan
        pixel_distance = calculate_2d_distance(coords[p1_idx], coords[p2_idx])
        return pixel_distance * scale_factor  # convertir a cm
    
    # Definir medidas a calcular (mismas que trial2.py)
    measurement_definitions = [
        # Cabeza y cuello
        ("Ancho cara (ojo_izq - ojo_der)", 1, 2, "7-10 cm"),
        # Torso  
        ("Ancho hombros", 5, 6, "35-45 cm"),
        ("Alto torso (hombro_izq - cadera_izq)", 5, 11, "50-70 cm"),
        ("Alto torso (hombro_der - cadera_der)", 6, 12, "50-70 cm"),
        ("Ancho caderas", 11, 12, "25-35 cm"),
        # Brazo izquierdo
        ("Brazo izq (hombro-codo)", 5, 7, "28-36 cm"),
        ("Antebrazo izq (codo-muñeca)", 7, 9, "23-30 cm"),
        ("Brazo completo izq (hombro-muñeca)", 5, 9, "55-70 cm"),
        # Brazo derecho
        ("Brazo der (hombro-codo)", 6, 8, "28-36 cm"),
        ("Antebrazo der (codo-muñeca)", 8, 10, "23-30 cm"),
        ("Brazo completo der (hombro-muñeca)", 6, 10, "55-70 cm"),
        # Pierna izquierda
        ("Muslo izq (cadera-rodilla)", 11, 13, "35-50 cm"),
        ("Pantorrilla izq (rodilla-tobillo)", 13, 15, "35-45 cm"),
        ("Pierna completa izq (cadera-tobillo)", 11, 15, "75-100 cm"),
        # Pierna derecha
        ("Muslo der (cadera-rodilla)", 12, 14, "35-50 cm"),
        ("Pantorrilla der (rodilla-tobillo)", 14, 16, "35-45 cm"),
        ("Pierna completa der (cadera-tobillo)", 12, 16, "75-100 cm"),
        # Medidas adicionales
        ("Estatura aprox (cabeza-tobillo_izq)", 0, 15, "150-190 cm"),
        ("Estatura aprox (cabeza-tobillo_der)", 0, 16, "150-190 cm"),
        ("Envergadura (muñeca_izq - muñeca_der)", 9, 10, "150-180 cm"),
    ]
    
    for cam_name, coords in cameras_data.items():
        if scale_factors[cam_name] is None:
            print(f"\n❌ Saltando {cam_name} (no se pudo calcular factor de escala)")
            continue
            
        scale_factor = scale_factors[cam_name]
        
        print(f"\n{'='*80}")
        print(f"ANÁLISIS DE MEDIDAS 2D ESCALADAS - {cam_name.upper()}")
        print(f"Factor de escala: {scale_factor:.4f} cm/pixel")
        print(f"{'='*80}")
        
        print(f"{'Medida':<40} | {'Valor':<12} | {'Rango Normal':<15} | {'Estado'}")
        print("-" * 85)
        
        valid_measurements = 0
        realistic_measurements = 0
        
        for name, p1_idx, p2_idx, normal_range in measurement_definitions:
            value_cm = distance_2d_scaled(coords, p1_idx, p2_idx, scale_factor)
            
            if np.isnan(value_cm):
                status = "❌ N/A"
                value_str = "N/A"
            else:
                value_str = f"{value_cm:.1f} cm"
                
                # Análisis básico de realismo (mismos rangos que trial2.py)
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
                    realistic = True
                    
                status = "✅ OK" if realistic else "⚠️ Fuera rango"
                if realistic:
                    realistic_measurements += 1
                valid_measurements += 1
            
            print(f"{name:<40} | {value_str:<12} | {normal_range:<15} | {status}")
        
        # Resumen por cámara
        print(f"\n{'='*80}")
        print(f"RESUMEN DE VALIDACIÓN 2D - {cam_name.upper()}")
        print(f"{'='*80}")
        print(f"Medidas válidas: {valid_measurements}/{len(measurement_definitions)}")
        if valid_measurements > 0:
            percentage = (realistic_measurements / valid_measurements) * 100
            print(f"Medidas realistas: {realistic_measurements}/{valid_measurements} ({percentage:.1f}%)")
            
            if percentage > 80:
                print("🟢 EXCELENTE: Las proporciones 2D son muy realistas")
            elif percentage > 60:
                print("🟡 BUENO: Las proporciones 2D son aceptables") 
            else:
                print("🔴 PROBLEMA: Las proporciones 2D parecen irrealistas")
        else:
            print("❌ No hay medidas válidas para evaluar")


def calculate_ojos_nariz_distance_2d(valid_mask: np.ndarray, scale_factors: Dict[str, float]):
    """Calcula específicamente la distancia ojos-nariz para cada cámara."""
    
    cameras_data = {
        "camera0": coordinates_camera_0,
        "camera1": coordinates_camera_1, 
        "camera2": coordinates_camera_2
    }
    
    print(f"\n{'='*60}")
    print("ANÁLISIS ESPECÍFICO: DISTANCIA OJOS-NARIZ 2D")
    print(f"{'='*60}")
    
    for cam_name, coords in cameras_data.items():
        if scale_factors[cam_name] is None:
            continue
            
        scale_factor = scale_factors[cam_name]
        
        # 0=Nariz, 1=Ojo_izq, 2=Ojo_der
        distances = []
        
        if valid_mask[0] and valid_mask[1]:  # Nariz y Ojo_izq
            dist_nariz_ojo_izq = calculate_2d_distance(coords[0], coords[1]) * scale_factor
            distances.append(("Nariz-Ojo_izq", dist_nariz_ojo_izq))
        
        if valid_mask[0] and valid_mask[2]:  # Nariz y Ojo_der  
            dist_nariz_ojo_der = calculate_2d_distance(coords[0], coords[2]) * scale_factor
            distances.append(("Nariz-Ojo_der", dist_nariz_ojo_der))
        
        if distances:
            avg_distance = np.mean([dist for _, dist in distances])
            print(f"\n{cam_name}:")
            for name, dist in distances:
                print(f"  {name}: {dist:.1f} cm")
            print(f"  Promedio: {avg_distance:.1f} cm (rango normal: 2-4 cm)")
            
            status = "✅ OK" if 2 <= avg_distance <= 4 else "⚠️ Fuera rango"
            print(f"  Estado: {status}")


def main():
    """Función principal de análisis 2D."""
    print("=== ANÁLISIS DE MEDIDAS CORPORALES 2D ===")
    print("Basado en keypoints 2D con antebrazo de referencia = 30 cm")
    print()
    
    # Filtrar keypoints válidos
    valid_mask = filter_valid_keypoints(confidence_threshold=0.5)
    
    # Calcular factores de escala por cámara
    scale_factors = calculate_scale_factor_from_2d_forearm(valid_mask, real_forearm_length_cm=30.0)
    
    # Análisis de medidas corporales por cámara
    analyze_2d_body_measurements(valid_mask, scale_factors)
    
    # Análisis específico ojos-nariz
    calculate_ojos_nariz_distance_2d(valid_mask, scale_factors)
    
    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN GENERAL DEL ANÁLISIS 2D")
    print(f"{'='*80}")
    
    valid_cameras = [cam for cam, factor in scale_factors.items() if factor is not None]
    print(f"Cámaras con factor de escala válido: {len(valid_cameras)}/3")
    for cam in valid_cameras:
        print(f"  ✅ {cam}: Factor = {scale_factors[cam]:.4f} cm/pixel")
    
    invalid_cameras = [cam for cam, factor in scale_factors.items() if factor is None]
    for cam in invalid_cameras:
        print(f"  ❌ {cam}: No se pudo calcular factor de escala")
    
    print(f"\n📊 Este análisis 2D permite evaluar la consistencia de las medidas")
    print(f"📊 sin necesidad de reconstrucción 3D completa")
    print(f"📊 Útil para validación rápida de proporciones corporales")


if __name__ == "__main__":
    main()

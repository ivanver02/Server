import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Asegurar que config esté en el path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Datos de ejemplo
PATIENT_ID = 57
SESSION_ID = 57
CHUNK_ID = 6
FRAME_ID = 127
PERSON_HEIGHT_CM = 190.0

# Nombres de los keypoints según COCO pose (los extra no los muestro)
KEYPOINT_NAMES = [
    "Nariz",           # 0
    "Ojo izquierdo",   # 1
    "Ojo derecho",     # 2
    "Oreja izquierda", # 3
    "Oreja derecha",   # 4
    "Hombro izquierdo", # 5
    "Hombro derecho",  # 6
    "Codo izquierdo",  # 7
    "Codo derecho",    # 8
    "Muñeca izquierda", # 9
    "Muñeca derecha",  # 10
    "Cadera izquierda", # 11
    "Cadera derecha",  # 12
    "Rodilla izquierda", # 13
    "Rodilla derecha", # 14
    "Tobillo izquierdo", # 15
    "Tobillo derecho"  # 16
]


def load_3d_reconstruction(patient_id: str, session_id: str, frame_id: int, chunk_id: int) -> Optional[np.ndarray]:
    """
    Carga una reconstrucción 3D ya procesada desde el archivo correspondiente.
    Formato: Server/data/processed/3D_keypoints/patient{id}/session{id}/{frame_id}_{chunk_id}.npy
    """
    # Construir ruta del archivo
    reconstruction_file = _ROOT / "data" / "processed" / "3D_keypoints" / f"patient{patient_id}" / f"session{session_id}" / f"{frame_id}_{chunk_id}.npy"
    
    if not reconstruction_file.exists():
        logger.error(f"Archivo de reconstrucción 3D no encontrado: {reconstruction_file}")
        return None
    
    try:
        points_3d = np.load(reconstruction_file)
        logger.info(f"Reconstrucción 3D cargada: {points_3d.shape} puntos desde {reconstruction_file}")
        return points_3d
    except Exception as e:
        logger.error(f"Error cargando reconstrucción 3D: {e}")
        return None


def check_3d_reconstruction_validity(points_3d: np.ndarray) -> Dict[str, any]:
    """
    Analiza la validez de la reconstrucción 3D cargada.
    """
    if points_3d is None:
        return {'valid': False, 'reason': 'Archivo no cargado'}
    
    # Contar puntos válidos (no NaN)
    valid_points = np.sum(~np.isnan(points_3d[:, 0]))
    total_points = len(points_3d)
    
    # Verificar dimensiones
    if points_3d.shape[1] != 3:
        return {'valid': False, 'reason': f'Dimensiones incorrectas: {points_3d.shape}'}
    
    # Verificar que hay suficientes puntos válidos
    if valid_points < 5:
        return {'valid': False, 'reason': f'Pocos puntos válidos: {valid_points}/{total_points}'}
    
    # Calcular estadísticas básicas
    valid_coords = points_3d[~np.isnan(points_3d).any(axis=1)]
    if len(valid_coords) == 0:
        return {'valid': False, 'reason': 'No hay coordenadas válidas'}
    
    # Rango de coordenadas (en metros, ya escaladas)
    coord_ranges = {
        'x_range': (np.min(valid_coords[:, 0]), np.max(valid_coords[:, 0])),
        'y_range': (np.min(valid_coords[:, 1]), np.max(valid_coords[:, 1])),
        'z_range': (np.min(valid_coords[:, 2]), np.max(valid_coords[:, 2]))
    }
    
    return {
        'valid': True,
        'total_points': total_points,
        'valid_points': valid_points,
        'completeness': (valid_points / total_points) * 100,
        'coordinate_ranges': coord_ranges
    }


def calculate_scale_factor_from_height(points_3d: np.ndarray, person_height_cm: float):
    """
    Calcula el factor de escala basado en la altura de la persona (nariz a tobillos).
    NOTA: Si los puntos ya están escalados, este factor debería estar cerca de 1.0
    """
    
    # Índices: 0=Nariz, 15=Tobillo_izq, 16=Tobillo_der
    target_distance_cm = person_height_cm - 15.0  # Altura menos 15cm
    height_measurements = []
    
    # Distancia nariz a tobillo izquierdo
    if not np.isnan(points_3d[0, 0]) and not np.isnan(points_3d[15, 0]):
        nose_to_left_ankle = np.linalg.norm(points_3d[0] - points_3d[15])
        height_measurements.append(("Nariz-Tobillo izquierdo", nose_to_left_ankle))
    
    # Distancia nariz a tobillo derecho
    if not np.isnan(points_3d[0, 0]) and not np.isnan(points_3d[16, 0]):
        nose_to_right_ankle = np.linalg.norm(points_3d[0] - points_3d[16])
        height_measurements.append(("Nariz-Tobillo derecho", nose_to_right_ankle))
    
    if not height_measurements:
        logger.warning("No se pueden calcular medidas nariz-tobillos")
        return None
    
    # Usar el promedio de las medidas disponibles
    avg_height_distance_m = np.mean([distance for _, distance in height_measurements])
    target_distance_m = target_distance_cm / 100.0  # convertir a metros
    
    # Factor de escala
    scale_factor = target_distance_m / avg_height_distance_m
    
    return scale_factor, height_measurements, avg_height_distance_m, target_distance_m


def display_3d_keypoints(points_3d: np.ndarray):
    """
    Muestra las coordenadas 3D de todos los keypoints detectados.
    """
    print(f"\n{'='*70}")
    print("COORDENADAS 3D DE LOS KEYPOINTS")
    print(f"{'='*70}")
    
    print(f"{'Keypoint':<20} | {'Coordenadas (x, y, z) en metros':<35} | {'Estado'}")
    print("-" * 70)
    
    valid_count = 0
    for i, name in enumerate(KEYPOINT_NAMES):
        if i < len(points_3d):
            x, y, z = points_3d[i]
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                coord_str = "N/A"
                status = "No detectado"
            else:
                coord_str = f"({x:.3f}, {y:.3f}, {z:.3f})"
                status = "Detectado"
                valid_count += 1
            
            print(f"{name:<20} | {coord_str:<35} | {status}")
        else:
            print(f"{name:<20} | {'N/A':<35} | {'Fuera de rango'}")
    
    print(f"\nResumen: {valid_count}/{len(KEYPOINT_NAMES)} keypoints detectados ({(valid_count/len(KEYPOINT_NAMES)*100):.1f}%)")
    print(f"{'='*70}")


def analyze_body_measurements_scaled(points_3d: np.ndarray, method_name: str, person_height_cm: float):
    """Analiza las medidas corporales de una reconstrucción 3D ya procesada"""
    
    # Los puntos ya deberían estar escalados, pero verificamos el factor de escala
    scale_result = calculate_scale_factor_from_height(points_3d, person_height_cm)
    if scale_result is None:
        logger.error("No se pudo verificar el factor de escala")
        return
    
    scale_factor, height_measurements, avg_height_m, target_height_m = scale_result
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE RECONSTRUCCIÓN 3D ALMACENADA ({method_name})")
    print(f"Archivo: patient{PATIENT_ID}/session{SESSION_ID}/{FRAME_ID}_{CHUNK_ID}.npy")
    print(f"Altura esperada: {person_height_cm} cm")
    print(f"{'='*70}")
    
    # Información sobre el escalado
    print(f"\n=== VERIFICACIÓN DE ESCALA")
    print(f"Altura objetivo (altura - 15cm): {(person_height_cm - 15):.1f} cm")
    for name, distance in height_measurements:
        print(f"{name}: {distance*100:.1f} cm (almacenado)")
    print(f"Altura promedio medida: {avg_height_m*100:.1f} cm")
    print(f"Factor de escala calculado: {scale_factor:.4f}")
    
    if abs(scale_factor - 1.0) < 0.05:
        print("✓ Los puntos parecen estar correctamente escalados")
        scaled_points = points_3d
    else:
        print(f"⚠ Los puntos pueden necesitar reescalado (factor: {scale_factor:.3f})")
        scaled_points = points_3d  # Mantenemos los puntos como están almacenados
    
    def distance_3d(p1_idx: int, p2_idx: int) -> float:
        if np.isnan(scaled_points[p1_idx, 0]) or np.isnan(scaled_points[p2_idx, 0]):
            return np.nan
        return np.linalg.norm(scaled_points[p1_idx] - scaled_points[p2_idx]) * 100  # convertir a cm
    
    def calculate_knee_angle(hip_idx: int, knee_idx: int, ankle_idx: int) -> float:
        """Calcula el ángulo de doblez de la rodilla en grados"""
        if (np.isnan(scaled_points[hip_idx, 0]) or 
            np.isnan(scaled_points[knee_idx, 0]) or 
            np.isnan(scaled_points[ankle_idx, 0])):
            return np.nan
        
        # Vectores: cadera->rodilla y rodilla->tobillo
        vec_hip_knee = scaled_points[knee_idx] - scaled_points[hip_idx]
        vec_knee_ankle = scaled_points[ankle_idx] - scaled_points[knee_idx]
        
        # Calcular ángulo entre vectores
        cos_angle = np.dot(vec_hip_knee, vec_knee_ankle) / (
            np.linalg.norm(vec_hip_knee) * np.linalg.norm(vec_knee_ankle)
        )
        
        # Asegurar que cos_angle esté en el rango válido [-1, 1]
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Ángulo en radianes y luego a grados
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # El ángulo de doblez es 180° - ángulo entre vectores
        knee_bend_angle = 180.0 - angle_deg
        
        return knee_bend_angle

    # ÁNGULOS DE DOBLEZ DE RODILLAS
    print(f"\n{'='*50}")
    print("ÁNGULOS DE DOBLEZ DE RODILLAS")
    print(f"{'='*50}")
    
    # Calcular ángulos de rodillas (índices: 11=cadera_izq, 13=rodilla_izq, 15=tobillo_izq, 12=cadera_der, 14=rodilla_der, 16=tobillo_der)
    left_knee_angle = calculate_knee_angle(11, 13, 15)  # cadera_izq -> rodilla_izq -> tobillo_izq
    right_knee_angle = calculate_knee_angle(12, 14, 16)  # cadera_der -> rodilla_der -> tobillo_der
    
    print(f"{'Articulación':<20} | {'Ángulo de Doblez':<15} | {'Estado'}")
    print("-" * 50)
    
    # Rodilla izquierda
    if np.isnan(left_knee_angle):
        left_status = "N/A"
        left_angle_str = "N/A"
    else:
        left_angle_str = f"{left_knee_angle:.1f}°"
        if 0 <= left_knee_angle <= 10:
            left_status = "Recta"
        elif 10 < left_knee_angle <= 30:
            left_status = "Poco doblada"
        elif 30 < left_knee_angle <= 90:
            left_status = "Doblada"
        elif 90 < left_knee_angle <= 140:
            left_status = "Muy doblada"
        else:
            left_status = "Extrema"
    
    print(f"{'Rodilla izquierda':<20} | {left_angle_str:<15} | {left_status}")
    
    # Rodilla derecha
    if np.isnan(right_knee_angle):
        right_status = "N/A"
        right_angle_str = "N/A"
    else:
        right_angle_str = f"{right_knee_angle:.1f}°"
        if 0 <= right_knee_angle <= 10:
            right_status = "Recta"
        elif 10 < right_knee_angle <= 30:
            right_status = "Poco doblada"
        elif 30 < right_knee_angle <= 90:
            right_status = "Doblada"
        elif 90 < right_knee_angle <= 140:
            right_status = "Muy doblada"
        else:
            right_status = "Extrema"
    
    print(f"{'Rodilla derecha':<20} | {right_angle_str:<15} | {right_status}")
    
    # Comparación entre rodillas
    if not np.isnan(left_knee_angle) and not np.isnan(right_knee_angle):
        angle_diff = abs(left_knee_angle - right_knee_angle)
        print(f"\nDiferencia entre rodillas: {angle_diff:.1f}°")
        if angle_diff <= 5:
            print("Simetría: Excelente (≤5°)")
        elif angle_diff <= 10:
            print("Simetría: Buena (≤10°)")
        elif angle_diff <= 15:
            print("Simetría: Regular (≤15°)")
        else:
            print("Simetría: Asimétrica (>15°)")

    # MEDIDAS CORPORALES
    print(f"\n{'='*50}")
    print("MEDIDAS CORPORALES")
    print(f"{'='*50}")

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
        ("Estatura aprox (nariz-tobillo_izq)", 
         distance_3d(0, 15) if not np.isnan(distance_3d(0, 15)) else np.nan, "150-190 cm"),
        ("Estatura aprox (nariz-tobillo_der)", 
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
            value_str = f"{value:.1f} cm"
            
            # Análisis de realismo basado en rangos
            if "cara" in name.lower():
                realistic = 5 <= value <= 15
            elif "ojos-nariz" in name.lower():
                realistic = 1 <= value <= 6
            elif "hombros" in name.lower():
                realistic = 25 <= value <= 55
            elif "torso" in name.lower():
                realistic = 40 <= value <= 80
            elif "caderas" in name.lower():
                realistic = 20 <= value <= 45
            elif "brazo" in name.lower() and "completo" not in name.lower():
                realistic = 20 <= value <= 45
            elif "brazo completo" in name.lower():
                realistic = 45 <= value <= 80
            elif "muslo" in name.lower():
                realistic = 30 <= value <= 60
            elif "pantorrilla" in name.lower():
                realistic = 30 <= value <= 55
            elif "pierna completa" in name.lower():
                realistic = 65 <= value <= 110
            elif "estatura" in name.lower():
                realistic = 140 <= value <= 200
            elif "envergadura" in name.lower():
                realistic = 140 <= value <= 200
            else:
                realistic = True
                
            status = "OK" if realistic else "Fuera rango"
            if realistic:
                realistic_measurements += 1
            valid_measurements += 1
        
        print(f"{name:<40} | {value_str:<12} | {normal_range:<15} | {status}")
    
    print(f"\nResumen: {realistic_measurements}/{valid_measurements} medidas realistas ({(realistic_measurements/valid_measurements*100):.1f}%)" if valid_measurements > 0 else "\nNo se pudieron calcular medidas")


def main():
    """
    Función principal que analiza una reconstrucción 3D ya almacenada.
    """
    
    print("� ANÁLISIS DE RECONSTRUCCIÓN 3D ALMACENADA")
    print("="*80)
    print(f"Paciente: {PATIENT_ID}, Sesión: {SESSION_ID}")
    print(f"Frame: {FRAME_ID}, Chunk: {CHUNK_ID}")
    print(f"Altura persona: {PERSON_HEIGHT_CM} cm")
    print()
    
    try:
        # 1. Cargar reconstrucción 3D almacenada
        logger.info("Cargando reconstrucción 3D almacenada...")
        points_3d = load_3d_reconstruction(PATIENT_ID, SESSION_ID, FRAME_ID, CHUNK_ID)
        
        if points_3d is None:
            logger.error("No se pudo cargar la reconstrucción 3D")
            print("❌ Error: No se encontró el archivo de reconstrucción 3D")
            return
        
        print(f"✓ Reconstrucción 3D cargada exitosamente")
        
        # 2. Verificar validez de la reconstrucción
        logger.info("Verificando validez de la reconstrucción...")
        validity_info = check_3d_reconstruction_validity(points_3d)
        
        if not validity_info['valid']:
            logger.error(f"Reconstrucción 3D inválida: {validity_info['reason']}")
            print(f"❌ Error: {validity_info['reason']}")
            return
        
        print(f"\n=== INFORMACIÓN DE LA RECONSTRUCCIÓN")
        print(f"Total de keypoints: {validity_info['total_points']}")
        print(f"Keypoints válidos: {validity_info['valid_points']} ({validity_info['completeness']:.1f}%)")
        
        coord_ranges = validity_info['coordinate_ranges']
        print(f"Rango X: {coord_ranges['x_range'][0]:.3f} a {coord_ranges['x_range'][1]:.3f} m")
        print(f"Rango Y: {coord_ranges['y_range'][0]:.3f} a {coord_ranges['y_range'][1]:.3f} m") 
        print(f"Rango Z: {coord_ranges['z_range'][0]:.3f} a {coord_ranges['z_range'][1]:.3f} m")
        
        # 3. Mostrar coordenadas de todos los keypoints
        logger.info("Mostrando coordenadas de keypoints...")
        display_3d_keypoints(points_3d)
        
        # 4. Análisis de medidas corporales
        logger.info("Ejecutando análisis de medidas corporales...")
        analyze_body_measurements_scaled(points_3d, "Reconstrucción_Almacenada", PERSON_HEIGHT_CM)
        
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETO FINALIZADO")
        print("Reconstrucción 3D cargada desde archivo y analizada exitosamente")
        print("="*80)
        
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        print(f"❌ Error: {e}")
    except Exception as e:
        logger.error(f"Error en el análisis: {e}")
        print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    main()
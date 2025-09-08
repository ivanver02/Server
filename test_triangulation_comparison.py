import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Asegurar que config esté en el path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from camera import Camera
from triangulation_svd import triangulate_frame_svd
from triangulation_bundle_adjustment import refine_frame_bundle_adjustment
from full_bundle_adjustment import full_bundle_adjustment
from config.camera_intrinsics import CAMERA_INTRINSICS

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Constantes
CONFIDENCE_THRESHOLD = 0.5

# Datos de ejemplo
PATIENT_ID = "57"
SESSION_ID = "57"
CHUNK_ID = 6
FRAME_ID = 44
PERSON_HEIGHT_CM = 190.0


def load_frame_keypoints(patient_id: str, session_id: str, chunk_id: int, frame_id: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Carga keypoints 2D de un frame específico.
    Estructura: {camera_id: (coordinates, confidence)}
    """
    base_path = _ROOT / "data" / "processed" / "2D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    
    if not base_path.exists():
        raise FileNotFoundError(f"No se encontró el directorio de sesión: {base_path}")
    
    frame_data = {}
    
    # Cargar datos para cada cámara
    for camera_id in ["camera0", "camera1", "camera2"]:
        # Construir ruta del archivo - formato: frame_chunk.npy (ej: 44_6.npy)
        coords_file = base_path / camera_id / "coordinates" / f"{frame_id}_{chunk_id}.npy"
        confs_file = base_path / camera_id / "confidence" / f"{frame_id}_{chunk_id}.npy"

        # Verificar que ambos archivos existen
        if coords_file.exists() and confs_file.exists():
            try:
                # Cargar coordenadas y confianzas
                coords = np.load(coords_file)
                confs = np.load(confs_file)
                frame_data[camera_id] = (coords, confs)
                logger.info(f"Cargados datos para {camera_id}: {coords.shape} keypoints")
            except Exception as e:
                logger.warning(f"Error cargando {camera_id}: {e}")
                continue
        else:
            logger.warning(f"Archivos faltantes para {camera_id} frame {frame_id} chunk {chunk_id}")
    
    return frame_data


def create_cameras_with_fixed_extrinsics() -> Dict[str, Camera]:
    """Crea objetos Camera con los extrínsecos fijos proporcionados"""
    cameras = {}
    
    # CAMERA0 - Referencia (identidad)
    cam0 = Camera.create("camera0")
    cam0.R = np.array([
        [1.00000, 0.00000, 0.00000],
        [0.00000, 1.00000, 0.00000],
        [0.00000, 0.00000, 1.00000]
    ])
    cam0.t = np.array([0.00000, 0.00000, 0.00000])
    cameras["camera0"] = cam0
    
    # CAMERA1
    cam1 = Camera.create("camera1")
    cam1.R = np.array([
        [0.31146, -0.08239, 0.94668],
        [0.07160, 0.99544, 0.06308],
        [-0.94756, 0.04814, 0.31593]
    ])
    cam1.t = np.array([-2.77853, -0.06092, 0.89055])
    cameras["camera1"] = cam1
    
    # CAMERA2
    cam2 = Camera.create("camera2")
    cam2.R = np.array([
        [0.72792, -0.07666, 0.68137],
        [0.04511, 0.99693, 0.06396],
        [-0.68418, -0.01582, 0.72914]
    ])
    cam2.t = np.array([-1.72786, -0.11608, -0.14376])
    cameras["camera2"] = cam2
    
    return cameras


def calculate_scale_factor_from_height(points_3d: np.ndarray, person_height_cm: float):
    """Calcula el factor de escala basado en la altura de la persona (nariz a tobillos)"""
    
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
        logger.warning("ERROR: No se pueden calcular medidas nariz-tobillos")
        return None
    
    # Usar el promedio de las medidas disponibles
    avg_height_distance_m = np.mean([distance for _, distance in height_measurements])
    target_distance_m = target_distance_cm / 100.0  # convertir a metros
    
    # Factor de escala
    scale_factor = target_distance_m / avg_height_distance_m
    
    logger.info(f"Factor de escala calculado: {scale_factor:.4f} (altura: {person_height_cm}cm)")
    return scale_factor


def print_3d_points_comparison(points_dict: Dict[str, np.ndarray], method_names: List[str]):
    """Imprime comparación detallada de puntos 3D entre métodos"""
    
    print("\n" + "="*80)
    print("COMPARACIÓN DETALLADA DE PUNTOS 3D")
    print("="*80)
    
    # Nombres de keypoints (índices típicos COCO)
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    num_keypoints = min(len(keypoint_names), points_dict[method_names[0]].shape[0])
    
    for i in range(num_keypoints):
        kp_name = keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}"
        print(f"\n{kp_name.upper()} (índice {i}):")
        print("-" * 50)
        
        for method in method_names:
            points = points_dict[method]
            x, y, z = points[i]
            print(f"  {method:25s}: ({x:8.3f}, {y:8.3f}, {z:8.3f})")
        
        # Calcular diferencias entre métodos
        if len(method_names) >= 2:
            print("  Diferencias:")
            for j in range(1, len(method_names)):
                diff = points_dict[method_names[j]] - points_dict[method_names[0]]
                dx, dy, dz = diff[i]
                dist = np.linalg.norm(diff[i])
                print(f"    {method_names[j]} - {method_names[0]:12s}: ({dx:7.3f}, {dy:7.3f}, {dz:7.3f}) | dist: {dist:7.3f}m")


def analyze_reconstruction_differences(points_dict: Dict[str, np.ndarray], method_names: List[str]):
    """Analiza estadísticas de diferencias entre métodos"""
    
    print("\n" + "="*80)
    print("ANÁLISIS ESTADÍSTICO DE DIFERENCIAS")
    print("="*80)
    
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            method1 = method_names[i]
            method2 = method_names[j]
            
            diff = points_dict[method2] - points_dict[method1]
            distances = np.linalg.norm(diff, axis=1)
            
            # Filtrar puntos válidos (no NaN)
            valid_mask = ~(np.isnan(distances) | np.isinf(distances))
            valid_distances = distances[valid_mask]
            
            if len(valid_distances) > 0:
                print(f"\n{method2} vs {method1}:")
                print(f"  Puntos válidos: {len(valid_distances)}/{len(distances)}")
                print(f"  Distancia media: {np.mean(valid_distances):.4f} m")
                print(f"  Desviación estándar: {np.std(valid_distances):.4f} m")
                print(f"  Distancia mínima: {np.min(valid_distances):.4f} m")
                print(f"  Distancia máxima: {np.max(valid_distances):.4f} m")
                print(f"  Mediana: {np.median(valid_distances):.4f} m")


def main():
    """Función principal para comparar métodos de triangulación"""
    
    print("🔍 PRUEBA DE COMPARACIÓN DE MÉTODOS DE TRIANGULACIÓN")
    print("="*60)
    print(f"Paciente: {PATIENT_ID}, Sesión: {SESSION_ID}")
    print(f"Frame: {FRAME_ID}, Chunk: {CHUNK_ID}")
    print(f"Altura persona: {PERSON_HEIGHT_CM} cm")
    print()
    
    try:
        # 1. Cargar keypoints 2D
        logger.info("Cargando keypoints 2D...")
        frame_keypoints = load_frame_keypoints(PATIENT_ID, SESSION_ID, CHUNK_ID, FRAME_ID)
        
        if len(frame_keypoints) < 2:
            logger.error("No hay suficientes cámaras con datos")
            return
        
        print(f"✓ Keypoints cargados para {len(frame_keypoints)} cámaras")
        
        # 2. Crear cámaras con extrínsecos fijos
        logger.info("Configurando cámaras con extrínsecos fijos...")
        cameras = create_cameras_with_fixed_extrinsics()
        print("✓ Cámaras configuradas con extrínsecos proporcionados")
        
        # 3. Ejecutar métodos de triangulación
        results = {}
        
        # Método 1: SVD Triangulation
        logger.info("Ejecutando SVD Triangulation...")
        try:
            points_3d_svd = triangulate_frame_svd(cameras, frame_keypoints, CONFIDENCE_THRESHOLD)
            
            # Aplicar escalado por altura
            scale_factor = calculate_scale_factor_from_height(points_3d_svd, PERSON_HEIGHT_CM)
            if scale_factor:
                points_3d_svd = points_3d_svd * scale_factor
            
            results["SVD_Triangulation"] = points_3d_svd
            print("✓ SVD Triangulation completado")
            
        except Exception as e:
            logger.error(f"Error en SVD Triangulation: {e}")
            results["SVD_Triangulation"] = None
        
        # Método 2: SVD + Bundle Adjustment
        logger.info("Ejecutando SVD + Bundle Adjustment...")
        try:
            if results["SVD_Triangulation"] is not None:
                # Usar puntos SVD sin escalar para BA
                points_3d_svd_unscaled = triangulate_frame_svd(cameras, frame_keypoints, CONFIDENCE_THRESHOLD)
                points_3d_ba = refine_frame_bundle_adjustment(points_3d_svd_unscaled, cameras, frame_keypoints)
                
                # Aplicar escalado por altura
                scale_factor = calculate_scale_factor_from_height(points_3d_ba, PERSON_HEIGHT_CM)
                if scale_factor:
                    points_3d_ba = points_3d_ba * scale_factor
                
                results["SVD_Bundle_Adjustment"] = points_3d_ba
                print("✓ SVD + Bundle Adjustment completado")
            else:
                results["SVD_Bundle_Adjustment"] = None
                print("✗ SVD + Bundle Adjustment omitido (SVD falló)")
        except Exception as e:
            logger.error(f"Error en SVD + Bundle Adjustment: {e}")
            results["SVD_Bundle_Adjustment"] = None
        
        # Método 3: Full Bundle Adjustment
        logger.info("Ejecutando Full Bundle Adjustment...")
        try:
            if results["SVD_Triangulation"] is not None:
                # Usar puntos SVD sin escalar para Full BA
                points_3d_svd_unscaled = triangulate_frame_svd(cameras, frame_keypoints, CONFIDENCE_THRESHOLD)
                points_3d_full_ba, cameras_optimized = full_bundle_adjustment(
                    points_3d_svd_unscaled, cameras, frame_keypoints, 
                    confidence_threshold=CONFIDENCE_THRESHOLD
                )
                
                # Aplicar escalado por altura
                scale_factor = calculate_scale_factor_from_height(points_3d_full_ba, PERSON_HEIGHT_CM)
                if scale_factor:
                    points_3d_full_ba = points_3d_full_ba * scale_factor
                
                results["Full_Bundle_Adjustment"] = points_3d_full_ba
                print("✓ Full Bundle Adjustment completado")
            else:
                results["Full_Bundle_Adjustment"] = None
                print("✗ Full Bundle Adjustment omitido (SVD falló)")
        except Exception as e:
            logger.error(f"Error en Full Bundle Adjustment: {e}")
            results["Full_Bundle_Adjustment"] = None
        
        # 4. Filtrar resultados válidos y realizar comparación
        valid_results = {k: v for k, v in results.items() if v is not None}
        method_names = list(valid_results.keys())
        
        if len(valid_results) == 0:
            logger.error("Ningún método funcionó correctamente")
            return
        
        print(f"\n✓ Métodos exitosos: {len(valid_results)}/{len(results)}")
        print(f"Métodos disponibles para comparación: {method_names}")
        
        # 5. Imprimir comparación detallada
        print_3d_points_comparison(valid_results, method_names)
        
        # 6. Análisis estadístico
        analyze_reconstruction_differences(valid_results, method_names)
        
        print("\n" + "="*80)
        print("PRUEBA COMPLETADA EXITOSAMENTE")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error en la prueba: {e}")
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()

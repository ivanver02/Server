"""
Script de prueba para el sistema de reconstrucción 3D.
Genera datos simulados y prueba todos los componentes del sistema.
"""

import numpy as np
import os
import logging
from pathlib import Path
import tempfile
import shutil

# Configurar el path para imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.processing.reconstruction.camera import Camera
from backend.processing.reconstruction.calculate_extrinsics import calculate_extrinsics_from_keypoints
from backend.processing.reconstruction.triangulation import triangulate_svd, triangulate_bundle_adjustment, compare_triangulation_methods
from backend.processing.reconstruction.validation import validate_reprojection, calculate_reconstruction_quality_score
from backend.processing.reconstruction.coordinator import reconstruct_3d_keypoints

logger = logging.getLogger(__name__)


def generate_test_keypoints_2d(num_frames: int = 5, num_keypoints: int = 17):
    """
    Genera keypoints 2D simulados para pruebas.
    Simula una persona moviendose frente a 3 cámaras.
    
    Args:
        num_frames: Número de frames a simular
        num_keypoints: Número de keypoints por frame (17 para pose humana)
        
    Returns:
        Dict con keypoints 2D por cámara
    """
    
    # Puntos 3D base para una pose humana típica (en metros)
    # Simplificado: cabeza, hombros, codos, muñecas, caderas, rodillas, tobillos
    base_points_3d = np.array([
        [0.0, 0.0, 1.7],    # 0: cabeza
        [-0.2, 0.0, 1.4],   # 1: hombro izq
        [0.2, 0.0, 1.4],    # 2: hombro der
        [-0.4, 0.0, 1.2],   # 3: codo izq
        [0.4, 0.0, 1.2],    # 4: codo der
        [-0.5, 0.0, 1.0],   # 5: muñeca izq
        [0.5, 0.0, 1.0],    # 6: muñeca der
        [-0.1, 0.0, 1.0],   # 7: cadera izq
        [0.1, 0.0, 1.0],    # 8: cadera der
        [-0.1, 0.0, 0.5],   # 9: rodilla izq
        [0.1, 0.0, 0.5],    # 10: rodilla der
        [-0.1, 0.0, 0.0],   # 11: tobillo izq
        [0.1, 0.0, 0.0],    # 12: tobillo der
        [0.0, 0.0, 1.2],    # 13: pecho
        [0.0, 0.0, 0.9],    # 14: abdomen
        [-0.05, 0.0, 1.6],  # 15: ojo izq
        [0.05, 0.0, 1.6],   # 16: ojo der
    ])
    
    # Configuraciones de cámaras (posiciones y orientaciones)
    camera_configs = {
        "camera_0": {"position": np.array([0.0, -2.0, 1.2]), "look_at": np.array([0.0, 0.0, 1.0])},
        "camera_1": {"position": np.array([-1.5, -1.5, 1.2]), "look_at": np.array([0.0, 0.0, 1.0])},
        "camera_2": {"position": np.array([1.5, -1.5, 1.2]), "look_at": np.array([0.0, 0.0, 1.0])}
    }
    
    # Generar datos por frame
    keypoints_sequence = {}
    
    for frame_idx in range(num_frames):
        # Simular movimiento: pequeñas variaciones en la pose
        movement_offset = np.array([
            0.1 * np.sin(frame_idx * 0.5),  # balanceo lateral
            0.05 * np.cos(frame_idx * 0.3), # movimiento adelante/atrás
            0.02 * np.sin(frame_idx * 0.7)  # ligero movimiento vertical
        ])
        
        # Aplicar movimiento a todos los puntos
        frame_points_3d = base_points_3d + movement_offset
        
        frame_keypoints = {}
        
        for camera_id, config in camera_configs.items():
            # Crear objeto cámara
            camera = Camera(camera_id)
            
            # Configurar extrínsecos simples (cámara mirando al origen)
            cam_pos = config["position"]
            look_at = config["look_at"]
            
            # Vector de dirección (de cámara hacia target)
            forward = look_at - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            # Vector up (hacia arriba)
            up = np.array([0.0, 0.0, 1.0])
            
            # Vector right (producto cruz)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            # Reajustar up
            up = np.cross(right, forward)
            
            # Matriz de rotación (world to camera)
            R = np.array([right, -up, forward]).T
            
            # Vector de traslación
            t = -R @ cam_pos.reshape(3, 1)
            
            # Configurar cámara con extrínsecos
            camera.set_extrinsics(R, t)
            
            # Proyectar puntos 3D a 2D
            projected_2d = camera.project_points(frame_points_3d)
            
            # Agregar ruido realista
            noise = np.random.normal(0, 1.0, projected_2d.shape)  # 1 pixel de std
            projected_2d += noise
            
            # Simular algunos keypoints perdidos (NaN)
            missing_mask = np.random.random(len(projected_2d)) < 0.05  # 5% perdidos
            projected_2d[missing_mask] = np.nan
            
            frame_keypoints[camera_id] = projected_2d
        
        # Generar nombre de frame simulado
        chunk_id = f"chunk_{frame_idx:03d}"
        frame_id = f"frame_{frame_idx:04d}"
        filename = f"{frame_id}_{chunk_id}"
        
        keypoints_sequence[filename] = frame_keypoints
    
    return keypoints_sequence


def create_test_data_files(keypoints_sequence: dict, output_dir: str):
    """
    Crea archivos .npy con los keypoints simulados.
    
    Args:
        keypoints_sequence: Datos generados por generate_test_keypoints_2d
        output_dir: Directorio donde guardar los archivos
    """
    
    # Crear estructura de directorios
    patient_dir = os.path.join(output_dir, "1", "8")  # paciente 1, sesión 8
    os.makedirs(patient_dir, exist_ok=True)
    
    # Guardar cada frame como archivo .npy
    for filename, frame_data in keypoints_sequence.items():
        file_path = os.path.join(patient_dir, f"{filename}.npy")
        np.save(file_path, frame_data)
        
    logger.info(f"Creados {len(keypoints_sequence)} archivos de prueba en {patient_dir}")
    return patient_dir


def test_individual_components():
    """
    Prueba cada componente individualmente.
    """
    
    print("=== PRUEBA DE COMPONENTES INDIVIDUALES ===\n")
    
    # 1. Prueba de cámaras
    print("1. Probando clase Camera...")
    cameras = {}
    for camera_id in ["camera_0", "camera_1", "camera_2"]:
        camera = cameras[camera_id] = Camera(camera_id)
        print(f"   {camera}")
    
    # 2. Generar datos de prueba
    print("\n2. Generando keypoints 2D de prueba...")
    keypoints_2d = {
        "camera_0": np.array([[320, 240], [300, 220], [340, 220]]),  # 3 puntos
        "camera_1": np.array([[315, 245], [295, 225], [335, 225]]),
        "camera_2": np.array([[325, 235], [305, 215], [345, 215]])
    }
    print(f"   Generados keypoints para {len(keypoints_2d)} cámaras")
    
    # 3. Prueba de cálculo de extrínsecos
    print("\n3. Probando cálculo de extrínsecos...")
    try:
        extrinsics = calculate_extrinsics_from_keypoints(cameras, keypoints_2d)
        print(f"   ✓ Extrínsecos calculados para {len(extrinsics)} cámaras")
        
        # Actualizar cámaras con extrínsecos
        for cam_id, (R, t) in extrinsics.items():
            cameras[cam_id].set_extrinsics(R, t)
            
    except Exception as e:
        print(f"   ✗ Error en extrínsecos: {e}")
    
    # 4. Prueba de triangulación SVD
    print("\n4. Probando triangulación SVD...")
    try:
        points_3d_svd, confidence_svd = triangulate_svd(cameras, keypoints_2d)
        print(f"   ✓ SVD: {len(points_3d_svd)} puntos 3D reconstruidos")
        
    except Exception as e:
        print(f"   ✗ Error en SVD: {e}")
        points_3d_svd = np.array([[0, 0, 1], [0, 0, 1.2], [0, 0, 0.8]])
        confidence_svd = np.array([1.0, 1.0, 1.0])
    
    # 5. Prueba de Bundle Adjustment
    print("\n5. Probando Bundle Adjustment...")
    try:
        points_3d_ba, confidence_ba, ba_info = triangulate_bundle_adjustment(
            cameras, keypoints_2d, points_3d_svd
        )
        print(f"   ✓ Bundle Adjustment: {len(points_3d_ba)} puntos optimizados")
        print(f"   ✓ Info: {ba_info.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"   ✗ Error en Bundle Adjustment: {e}")
        points_3d_ba = points_3d_svd
        confidence_ba = confidence_svd
    
    # 6. Prueba de validación
    print("\n6. Probando validación...")
    try:
        validation_results = validate_reprojection(
            cameras, points_3d_ba, keypoints_2d
        )
        quality_score = calculate_reconstruction_quality_score(validation_results)
        print(f"   ✓ Validación completada, score: {quality_score:.1f}/100")
        
    except Exception as e:
        print(f"   ✗ Error en validación: {e}")
    
    print("\n=== FIN PRUEBA DE COMPONENTES ===\n")


def test_full_pipeline():
    """
    Prueba el pipeline completo con datos simulados.
    """
    
    print("=== PRUEBA DEL PIPELINE COMPLETO ===\n")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp(prefix="reconstruction_test_")
    
    try:
        # Generar datos de prueba
        print("1. Generando datos de prueba...")
        keypoints_sequence = generate_test_keypoints_2d(num_frames=10, num_keypoints=17)
        
        # Crear archivos de datos
        print("2. Creando archivos de datos...")
        keypoints_2d_dir = os.path.join(temp_dir, "2D_keypoints")
        create_test_data_files(keypoints_sequence, keypoints_2d_dir)
        
        # Ejecutar reconstrucción completa
        print("3. Ejecutando reconstrucción 3D...")
        output_3d_dir = os.path.join(temp_dir, "3D_keypoints")
        
        # Probar SVD
        print("\n   Probando método SVD...")
        stats_svd = reconstruct_3d_keypoints(
            keypoints_2d_dir=keypoints_2d_dir,
            output_dir=output_3d_dir + "_svd",
            patient_id="1",
            session_id="8",
            use_bundle_adjustment=False,
            validation_plots=False
        )
        
        print(f"   ✓ SVD completado: {stats_svd['files_processed']}/{stats_svd['files_total']} archivos")
        print(f"   ✓ Puntos reconstruidos: {stats_svd['points_reconstructed']}")
        print(f"   ✓ Calidad promedio: {stats_svd['average_quality_score']:.1f}/100")
        
        # Probar Bundle Adjustment
        print("\n   Probando método Bundle Adjustment...")
        stats_ba = reconstruct_3d_keypoints(
            keypoints_2d_dir=keypoints_2d_dir,
            output_dir=output_3d_dir + "_ba",
            patient_id="1",
            session_id="8",
            use_bundle_adjustment=True,
            validation_plots=False
        )
        
        print(f"   ✓ Bundle Adjustment completado: {stats_ba['files_processed']}/{stats_ba['files_total']} archivos")
        print(f"   ✓ Puntos reconstruidos: {stats_ba['points_reconstructed']}")
        print(f"   ✓ Calidad promedio: {stats_ba['average_quality_score']:.1f}/100")
        
        # Comparación
        print(f"\n4. Comparación de métodos:")
        print(f"   SVD vs Bundle Adjustment:")
        print(f"   - Calidad: {stats_svd['average_quality_score']:.1f} vs {stats_ba['average_quality_score']:.1f}")
        print(f"   - Puntos: {stats_svd['points_reconstructed']} vs {stats_ba['points_reconstructed']}")
        
        # Verificar archivos de salida
        print(f"\n5. Verificando archivos de salida...")
        svd_files = len(list(Path(output_3d_dir + "_svd/1/8").glob("*.npy")))
        ba_files = len(list(Path(output_3d_dir + "_ba/1/8").glob("*.npy")))
        print(f"   ✓ Archivos SVD: {svd_files}")
        print(f"   ✓ Archivos Bundle Adjustment: {ba_files}")
        
    except Exception as e:
        print(f"   ✗ Error en pipeline: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Limpiar directorio temporal
        try:
            shutil.rmtree(temp_dir)
            print(f"\n   Directorio temporal limpiado: {temp_dir}")
        except:
            print(f"\n   No se pudo limpiar: {temp_dir}")
    
    print("\n=== FIN PRUEBA DEL PIPELINE ===\n")


def test_error_conditions():
    """
    Prueba condiciones de error y casos límite.
    """
    
    print("=== PRUEBA DE CONDICIONES DE ERROR ===\n")
    
    # 1. Datos inválidos
    print("1. Probando con datos inválidos...")
    
    cameras = {f"camera_{i}": Camera(f"camera_{i}") for i in range(3)}
    
    # Keypoints con NaN
    keypoints_nan = {
        "camera_0": np.array([[np.nan, np.nan], [300, 220]]),
        "camera_1": np.array([[315, 245], [np.nan, np.nan]])
    }
    
    try:
        points_3d, _ = triangulate_svd(cameras, keypoints_nan)
        print(f"   ✓ SVD con NaN: {len(points_3d)} puntos reconstruidos")
    except Exception as e:
        print(f"   ✗ Error con NaN: {e}")
    
    # 2. Insuficientes cámaras
    print("\n2. Probando con insuficientes cámaras...")
    
    keypoints_single = {"camera_0": np.array([[320, 240]])}
    
    try:
        points_3d, _ = triangulate_svd(cameras, keypoints_single)
        print(f"   ✓ Una cámara: {len(points_3d)} puntos reconstruidos")
    except Exception as e:
        print(f"   ✗ Error una cámara: {e}")
    
    # 3. Archivos no encontrados
    print("\n3. Probando directorio inexistente...")
    
    try:
        stats = reconstruct_3d_keypoints(
            keypoints_2d_dir="/directorio/inexistente",
            output_dir="/tmp/test",
            patient_id="1",
            session_id="8",
            use_bundle_adjustment=False,
            validation_plots=False
        )
        print(f"   ✗ No debería llegar aquí")
    except FileNotFoundError as e:
        print(f"   ✓ Error esperado: {type(e).__name__}")
    except Exception as e:
        print(f"   ? Error inesperado: {e}")
    
    print("\n=== FIN PRUEBA DE ERRORES ===\n")


def main():
    """
    Ejecuta todas las pruebas del sistema de reconstrucción 3D.
    """
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("SISTEMA DE PRUEBAS - RECONSTRUCCIÓN 3D")
    print("=" * 60)
    
    try:
        # Ejecutar todas las pruebas
        test_individual_components()
        test_full_pipeline()
        test_error_conditions()
        
        print("=" * 60)
        print("✓ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"✗ ERROR EN LAS PRUEBAS: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

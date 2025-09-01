"""
Coordinador principal del sistema de reconstrucción 3D.
Integra todos los módulos y procesa datos de una sesión completa.
"""

import numpy as np
import os
import glob
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .camera import Camera
from .calculate_extrinsics import calculate_extrinsics_from_keypoints
from .triangulation import triangulate_svd, triangulate_bundle_adjustment
from .validation import validate_reprojection, calculate_reconstruction_quality_score

logger = logging.getLogger(__name__)

def reconstruct_3d_keypoints(
    keypoints_2d_dir: str,
    output_dir: str,
    patient_id: str,
    session_id: str,
    use_bundle_adjustment: bool = True,
    validation_plots: bool = False
) -> Dict:
    """
    Reconstruye keypoints 3D para una sesión completa de paciente.
    
    Args:
        keypoints_2d_dir: Directorio con keypoints 2D procesados
        output_dir: Directorio base para guardar keypoints 3D
        patient_id: ID del paciente
        session_id: ID de la sesión
        use_bundle_adjustment: Si usar Bundle Adjustment (más lento pero preciso)
        validation_plots: Si generar plots de validación
        
    Returns:
        Dict con estadísticas del procesamiento
    """
    
    logger.info(f"Iniciando reconstrucción 3D para paciente {patient_id}, sesión {session_id}")
    
    # Crear directorio de salida
    session_output_dir = os.path.join(output_dir, patient_id, session_id)
    os.makedirs(session_output_dir, exist_ok=True)
    
    # Encontrar archivos de keypoints 2D
    keypoints_pattern = os.path.join(keypoints_2d_dir, patient_id, session_id, "*.npy")
    keypoint_files = glob.glob(keypoints_pattern)
    
    if not keypoint_files:
        raise FileNotFoundError(f"No se encontraron archivos de keypoints 2D en: {keypoints_pattern}")
    
    logger.info(f"Encontrados {len(keypoint_files)} archivos de keypoints 2D")
    
    # Inicializar cámaras
    cameras = {}
    for camera_id in ["camera_0", "camera_1", "camera_2"]:
        cameras[camera_id] = Camera(camera_id)
    
    # Estadísticas de procesamiento
    stats = {
        "patient_id": patient_id,
        "session_id": session_id,
        "files_processed": 0,
        "files_total": len(keypoint_files),
        "points_reconstructed": 0,
        "average_quality_score": 0.0,
        "method_used": "bundle_adjustment" if use_bundle_adjustment else "svd",
        "extrinsics_calculated": False,
        "validation_results": {}
    }
    
    extrinsics_calculated = False
    quality_scores = []
    
    for file_idx, keypoint_file in enumerate(sorted(keypoint_files)):
        try:
            # Extraer frame_id y chunk_id del nombre del archivo
            filename = Path(keypoint_file).stem
            # Formato esperado: {frame_id}_{chunk_id}
            
            logger.info(f"Procesando archivo {file_idx+1}/{len(keypoint_files)}: {filename}")
            
            # Cargar keypoints 2D
            keypoints_data = np.load(keypoint_file, allow_pickle=True).item()
            
            if not isinstance(keypoints_data, dict):
                logger.warning(f"Formato inválido en {filename}, saltando")
                continue
            
            # Verificar que tenemos datos de múltiples cámaras
            available_cameras = [cam for cam in ["camera_0", "camera_1", "camera_2"] if cam in keypoints_data]
            
            if len(available_cameras) < 2:
                logger.warning(f"Insuficientes cámaras en {filename} ({len(available_cameras)} < 2), saltando")
                continue
            
            # Calcular extrínsecos solo una vez (usar primer archivo válido)
            if not extrinsics_calculated:
                logger.info("Calculando parámetros extrínsecos...")
                
                keypoints_2d_for_extrinsics = {}
                for cam_id in available_cameras:
                    keypoints_2d_for_extrinsics[cam_id] = keypoints_data[cam_id]
                
                extrinsics = calculate_extrinsics_from_keypoints(
                    cameras, keypoints_2d_for_extrinsics
                )
                
                # Actualizar cámaras con extrínsecos calculados
                for cam_id, (R, t) in extrinsics.items():
                    cameras[cam_id].set_extrinsics(R, t)
                
                extrinsics_calculated = True
                stats["extrinsics_calculated"] = True
                logger.info("Parámetros extrínsecos calculados exitosamente")
            
            # Reconstruir puntos 3D
            if use_bundle_adjustment:
                # Usar Bundle Adjustment para máxima precisión
                points_3d, confidence, ba_info = triangulate_bundle_adjustment(
                    cameras, keypoints_data
                )
                reconstruction_method = "bundle_adjustment"
            else:
                # Usar SVD para velocidad
                points_3d, confidence = triangulate_svd(cameras, keypoints_data)
                reconstruction_method = "svd"
            
            if len(points_3d) == 0:
                logger.warning(f"No se pudieron reconstruir puntos 3D para {filename}")
                continue
            
            # Validar calidad de reconstrucción
            validation_results = validate_reprojection(
                cameras, points_3d, keypoints_data,
                save_plots=validation_plots,
                output_dir=session_output_dir if validation_plots else None
            )
            
            quality_score = calculate_reconstruction_quality_score(validation_results)
            quality_scores.append(quality_score)
            
            # Guardar keypoints 3D
            output_file = os.path.join(session_output_dir, f"{filename}.npy")
            
            # Estructura de datos para guardar
            output_data = {
                "points_3d": points_3d,
                "confidence": confidence,
                "reconstruction_method": reconstruction_method,
                "quality_score": quality_score,
                "validation_results": validation_results,
                "frame_info": {
                    "filename": filename,
                    "cameras_used": available_cameras,
                    "num_points": len(points_3d)
                }
            }
            
            if use_bundle_adjustment and 'ba_info' in locals():
                output_data["bundle_adjustment_info"] = ba_info
            
            np.save(output_file, output_data)
            
            stats["files_processed"] += 1
            stats["points_reconstructed"] += len(points_3d)
            
            logger.info(f"Archivo {filename} procesado: {len(points_3d)} puntos 3D, "
                       f"calidad: {quality_score:.1f}/100")
            
        except Exception as e:
            logger.error(f"Error procesando {keypoint_file}: {e}")
            continue
    
    # Estadísticas finales
    if quality_scores:
        stats["average_quality_score"] = float(np.mean(quality_scores))
        stats["min_quality_score"] = float(np.min(quality_scores))
        stats["max_quality_score"] = float(np.max(quality_scores))
    
    # Guardar estadísticas de la sesión
    stats_file = os.path.join(session_output_dir, "reconstruction_stats.npy")
    np.save(stats_file, stats)
    
    logger.info(f"Reconstrucción completada: {stats['files_processed']}/{stats['files_total']} archivos, "
               f"{stats['points_reconstructed']} puntos 3D, calidad promedio: {stats['average_quality_score']:.1f}/100")
    
    return stats


def test_reconstruction_with_sample_data():
    """
    Prueba el sistema de reconstrucción con datos del paciente 1, sesión 8.
    """
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Directorios de datos
    keypoints_2d_dir = r"c:\Users\Juan Cantizani\Server\data\processed\2D_keypoints"
    output_3d_dir = r"c:\Users\Juan Cantizani\Server\data\processed\3D_keypoints"
    
    patient_id = "1"
    session_id = "8"
    
    try:
        # Ejecutar reconstrucción con Bundle Adjustment
        logger.info("=== Prueba con Bundle Adjustment ===")
        stats_ba = reconstruct_3d_keypoints(
            keypoints_2d_dir=keypoints_2d_dir,
            output_dir=output_3d_dir,
            patient_id=patient_id,
            session_id=session_id,
            use_bundle_adjustment=True,
            validation_plots=True  # Generar plots para análisis
        )
        
        print(f"\n=== RESULTADOS BUNDLE ADJUSTMENT ===")
        print(f"Archivos procesados: {stats_ba['files_processed']}/{stats_ba['files_total']}")
        print(f"Puntos 3D reconstruidos: {stats_ba['points_reconstructed']}")
        print(f"Calidad promedio: {stats_ba['average_quality_score']:.1f}/100")
        print(f"Extrínsecos calculados: {stats_ba['extrinsics_calculated']}")
        
        # Ejecutar reconstrucción con SVD para comparación
        logger.info("\n=== Prueba con SVD ===")
        stats_svd = reconstruct_3d_keypoints(
            keypoints_2d_dir=keypoints_2d_dir,
            output_dir=output_3d_dir + "_svd",  # Directorio separado
            patient_id=patient_id,
            session_id=session_id,
            use_bundle_adjustment=False,
            validation_plots=False
        )
        
        print(f"\n=== RESULTADOS SVD ===")
        print(f"Archivos procesados: {stats_svd['files_processed']}/{stats_svd['files_total']}")
        print(f"Puntos 3D reconstruidos: {stats_svd['points_reconstructed']}")
        print(f"Calidad promedio: {stats_svd['average_quality_score']:.1f}/100")
        
        # Comparación
        print(f"\n=== COMPARACIÓN ===")
        print(f"Bundle Adjustment vs SVD:")
        print(f"  Calidad: {stats_ba['average_quality_score']:.1f} vs {stats_svd['average_quality_score']:.1f}")
        print(f"  Puntos: {stats_ba['points_reconstructed']} vs {stats_svd['points_reconstructed']}")
        
        return stats_ba, stats_svd
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
        raise


if __name__ == "__main__":
    # Ejecutar prueba con datos reales
    test_reconstruction_with_sample_data()

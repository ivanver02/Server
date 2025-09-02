"""
Prueba completa del sistema de reconstrucción 3D.
Lee datos reales del paciente 1, sesión 8, chunk 0.
"""

import numpy as np
import os
import glob
import logging
from pathlib import Path

# Imports de los módulos de reconstrucción
from backend.processing.reconstruction.camera import Camera
from backend.processing.reconstruction.calculate_extrinsics import calculate_extrinsics_from_keypoints
from backend.processing.reconstruction.triangulation_svd import triangulate_svd
from backend.processing.reconstruction.triangulation_bundle_adjustment import triangulate_bundle_adjustment
from backend.processing.reconstruction.reprojection import reproject_and_validate

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_keypoints_chunk(patient: str, session: str, chunk: str) -> dict:
    """
    Carga todos los frames de un chunk específico para todas las cámaras.
    
    Args:
        patient: ID del paciente
        session: ID de la sesión  
        chunk: ID del chunk
        
    Returns:
        Dict {frame_id: {camera_id: keypoints_2d}}
    """
    
    base_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "2D_keypoints"
    chunk_data = {}
    
    # Buscar archivos que contengan el chunk específico
    for camera_id in ["camera_0", "camera_1", "camera_2"]:
        camera_path = base_path / patient / session / camera_id / "coordinates"
        
        if not camera_path.exists():
            logger.warning(f"No existe directorio: {camera_path}")
            continue
        
        # Buscar archivos del chunk específico
        pattern = f"*_{chunk}.npy"
        chunk_files = list(camera_path.glob(pattern))
        
        logger.info(f"Cámara {camera_id}: {len(chunk_files)} frames encontrados para chunk {chunk}")
        
        for file_path in chunk_files:
            # Extraer frame_id del nombre del archivo
            filename = file_path.stem  # formato: frame_XXXX_chunk_XXX
            frame_id = filename.split('_')[1]  # extraer XXXX
            
            try:
                keypoints = np.load(file_path)
                
                if frame_id not in chunk_data:
                    chunk_data[frame_id] = {}
                
                chunk_data[frame_id][camera_id] = keypoints
                
            except Exception as e:
                logger.error(f"Error cargando {file_path}: {e}")
    
    logger.info(f"Cargados {len(chunk_data)} frames del chunk {chunk}")
    return chunk_data


def test_reconstruction_system():
    """
    Prueba completa del sistema de reconstrucción 3D.
    """
    
    print("=== PRUEBA SISTEMA RECONSTRUCCIÓN 3D ===\n")
    
    # Configuración
    patient_id = "1"
    session_id = "8" 
    chunk_id = "chunk_000"
    
    # 1. Cargar datos del chunk 0
    print("1. Cargando keypoints 2D del chunk 0...")
    chunk_data = load_keypoints_chunk(patient_id, session_id, chunk_id)
    
    if not chunk_data:
        print("   ✗ No se encontraron datos. Verificar ruta de datos.")
        return
    
    print(f"   ✓ Cargados {len(chunk_data)} frames")
    
    # 2. Inicializar cámaras
    print("\n2. Inicializando cámaras...")
    cameras = {}
    for camera_id in ["camera_0", "camera_1", "camera_2"]:
        cameras[camera_id] = Camera(camera_id)
        print(f"   ✓ {camera_id}: fx={cameras[camera_id].K[0,0]:.1f}, fy={cameras[camera_id].K[1,1]:.1f}")
    
    # 3. Procesar cada frame
    print(f"\n3. Procesando {len(chunk_data)} frames...")
    
    # Directorio de salida
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "3D_keypoints" / patient_id / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    svd_results = []
    ba_results = []
    
    for frame_id in sorted(chunk_data.keys()):
        frame_keypoints = chunk_data[frame_id]
        
        # Verificar que tenemos al menos 2 cámaras
        available_cameras = list(frame_keypoints.keys())
        if len(available_cameras) < 2:
            print(f"   Frame {frame_id}: Solo {len(available_cameras)} cámaras, saltando")
            continue
        
        print(f"   Frame {frame_id}: Cámaras {available_cameras}")
        
        # 4. Calcular extrínsecos
        extrinsics = calculate_extrinsics_from_keypoints(cameras, frame_keypoints)
        
        # Actualizar cámaras con extrínsecos
        for cam_id, (R, t) in extrinsics.items():
            cameras[cam_id].set_extrinsics(R, t)
        
        # 5. Reconstrucción con SVD
        points_3d_svd = triangulate_svd(cameras, frame_keypoints)
        
        # 6. Reconstrucción con Bundle Adjustment
        points_3d_ba = triangulate_bundle_adjustment(cameras, frame_keypoints, points_3d_svd)
        
        # 7. Validación por reproyección
        validation_svd = reproject_and_validate(cameras, points_3d_svd, frame_keypoints)
        validation_ba = reproject_and_validate(cameras, points_3d_ba, frame_keypoints)
        
        # 8. Guardar resultados
        frame_chunk_name = f"frame_{frame_id}_{chunk_id}"
        
        # Guardar SVD
        svd_file = output_dir / f"{frame_chunk_name}_svd.npy"
        np.save(svd_file, points_3d_svd)
        
        # Guardar Bundle Adjustment
        ba_file = output_dir / f"{frame_chunk_name}_ba.npy"
        np.save(ba_file, points_3d_ba)
        
        # Estadísticas
        svd_valid = np.sum(~np.isnan(points_3d_svd[:, 0])) if len(points_3d_svd) > 0 else 0
        ba_valid = np.sum(~np.isnan(points_3d_ba[:, 0])) if len(points_3d_ba) > 0 else 0
        
        svd_error = validation_svd.get("global", {}).get("mean_error", np.nan)
        ba_error = validation_ba.get("global", {}).get("mean_error", np.nan)
        
        svd_results.append({
            "frame": frame_id,
            "valid_points": svd_valid,
            "error": svd_error
        })
        
        ba_results.append({
            "frame": frame_id, 
            "valid_points": ba_valid,
            "error": ba_error
        })
        
        print(f"     SVD: {svd_valid} puntos, error {svd_error:.2f} px")
        print(f"     BA:  {ba_valid} puntos, error {ba_error:.2f} px")
    
    # 9. Resumen final
    print(f"\n=== RESUMEN FINAL ===")
    
    if svd_results:
        svd_total_points = sum(r["valid_points"] for r in svd_results)
        svd_errors = [r["error"] for r in svd_results if not np.isnan(r["error"])]
        svd_mean_error = np.mean(svd_errors) if svd_errors else np.nan
        
        ba_total_points = sum(r["valid_points"] for r in ba_results)
        ba_errors = [r["error"] for r in ba_results if not np.isnan(r["error"])]
        ba_mean_error = np.mean(ba_errors) if ba_errors else np.nan
        
        print(f"SVD:")
        print(f"  - Frames procesados: {len(svd_results)}")
        print(f"  - Puntos totales: {svd_total_points}")
        print(f"  - Error medio: {svd_mean_error:.2f} px")
        
        print(f"\nBundle Adjustment:")
        print(f"  - Frames procesados: {len(ba_results)}")
        print(f"  - Puntos totales: {ba_total_points}")
        print(f"  - Error medio: {ba_mean_error:.2f} px")
        
        if not np.isnan(svd_mean_error) and not np.isnan(ba_mean_error):
            improvement = ((svd_mean_error - ba_mean_error) / svd_mean_error) * 100
            print(f"\nMejora Bundle Adjustment: {improvement:.1f}%")
        
        print(f"\nArchivos guardados en: {output_dir}")
    
    print("\n=== FIN PRUEBA ===")


if __name__ == "__main__":
    test_reconstruction_system()

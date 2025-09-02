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
        patient: ID del paciente (ej: "1")
        session: ID de la sesión (ej: "8")
        chunk: ID del chunk (ej: "2")
        
    Returns:
        Dict {frame_id: {camera_id: keypoints_2d}}
    """
    
    base_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "2D_keypoints"
    chunk_data = {}
    
    # Formato de directorio: patient1/session8/camera0
    patient_dir = f"patient{patient}"
    session_dir = f"session{session}"
    
    # Buscar archivos que contengan el chunk específico
    for camera_num in [0, 1, 2]:
        camera_dir = f"camera{camera_num}"
        camera_path = base_path / patient_dir / session_dir / camera_dir / "coordinates"
        
        if not camera_path.exists():
            logger.warning(f"No existe directorio: {camera_path}")
            continue
        
        # Buscar archivos del chunk específico (formato: {frame_id}_{chunk_id}.npy)
        pattern = f"*_{chunk}.npy"
        chunk_files = list(camera_path.glob(pattern))
        
        logger.info(f"Cámara {camera_dir}: {len(chunk_files)} frames encontrados para chunk {chunk}")
        
        for file_path in chunk_files:
            # Extraer frame_id del nombre del archivo (formato: 3_2.npy)
            filename = file_path.stem  # ej: "3_2"
            frame_id = filename.split('_')[0]  # extraer "3"
            
            try:
                keypoints = np.load(file_path)
                
                if frame_id not in chunk_data:
                    chunk_data[frame_id] = {}
                
                chunk_data[frame_id][camera_dir] = keypoints
                
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
    chunk_id = "0"
    
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
    for camera_id in ["camera0", "camera1", "camera2"]:
        cameras[camera_id] = Camera(camera_id)
        print(f"   ✓ {camera_id}: fx={cameras[camera_id].K[0,0]:.1f}, fy={cameras[camera_id].K[1,1]:.1f}")
    
    # 3. Procesar cada frame
    print(f"\n3. Procesando {len(chunk_data)} frames...")
    
    # Directorio de salida

    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "3D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    svd_results = []
    ba_results = []
    
    # DEBUG: procesar solo el primer frame para análisis detallado
    frame_ids = sorted(chunk_data.keys())
    frame_ids = frame_ids[:1]  
    logger.info(f"PROCESANDO SOLO 1 FRAME PARA DEBUG: {frame_ids}")
    
    for frame_id in frame_ids:
        frame_keypoints = chunk_data[frame_id]
        
        # Verificar que tenemos al menos 2 cámaras
        available_cameras = list(frame_keypoints.keys())
        if len(available_cameras) < 2:
            print(f"   Frame {frame_id}: Solo {len(available_cameras)} cámaras, saltando")
            continue
        
        print(f"   Frame {frame_id}: Cámaras {available_cameras}")
        
        # Log de ejemplo de keypoints 2D para debug
        for cam_id in available_cameras[:2]:  # Solo mostrar 2 cámaras
            kpts = frame_keypoints[cam_id]
            valid_kpts = kpts[~np.isnan(kpts[:, 0])]
            logger.info(f"DEBUG - {cam_id}: {len(valid_kpts)} keypoints válidos")
            if len(valid_kpts) > 0:
                logger.info(f"  Ejemplo keypoint: {valid_kpts[0]}")
                logger.info(f"  Rango X: [{valid_kpts[:, 0].min():.1f}, {valid_kpts[:, 0].max():.1f}]")
                logger.info(f"  Rango Y: [{valid_kpts[:, 1].min():.1f}, {valid_kpts[:, 1].max():.1f}]")
        
        # 4. Calcular extrínsecos
        extrinsics = calculate_extrinsics_from_keypoints(cameras, frame_keypoints)
        
        # Actualizar cámaras con extrínsecos
        for cam_id, (R, t) in extrinsics.items():
            cameras[cam_id].set_extrinsics(R, t)
            # Log para verificar consistencia de matrices
            P = cameras[cam_id].get_projection_matrix()
            logger.info(f"Matriz proyección {cam_id}:")
            logger.info(f"  P shape: {P.shape}")
            logger.info(f"  P[:, 3] (columna traslación): {P[:, 3]}")
            logger.info(f"  Det(P[:, :3]): {np.linalg.det(P[:, :3]):.6f}")
        
        # 5. Reconstrucción con SVD
        points_3d_svd = triangulate_svd(cameras, frame_keypoints)
        
        # 6. Reconstrucción con Bundle Adjustment
        points_3d_ba = triangulate_bundle_adjustment(cameras, frame_keypoints, points_3d_svd)
        
        # 7. Validación por reproyección
        validation_svd = reproject_and_validate(cameras, points_3d_svd, frame_keypoints)
        validation_ba = reproject_and_validate(cameras, points_3d_ba, frame_keypoints)
        
        # VERIFICACIÓN MANUAL DE CONSISTENCIA
        # Tomar el primer punto y verificar manualmente la triangulación/reproyección
        logger.info("=== VERIFICACIÓN MANUAL ===")
        first_point_2d = {}
        for cam_id in ["camera0", "camera1", "camera2"]:
            if cam_id in frame_keypoints:
                pt = frame_keypoints[cam_id][0]
                if not np.isnan(pt).any():
                    first_point_2d[cam_id] = pt
                    logger.info(f"{cam_id} primer punto 2D: {pt}")
        
        if len(first_point_2d) >= 2:
            # Triangular manualmente usando DLT
            A_manual = []
            for cam_id, pt_2d in first_point_2d.items():
                camera = cameras[cam_id]
                P = camera.get_projection_matrix()
                x, y = pt_2d
                A_manual.append(x * P[2, :] - P[0, :])
                A_manual.append(y * P[2, :] - P[1, :])
            
            A_manual = np.array(A_manual)
            _, _, V = np.linalg.svd(A_manual)
            X_manual = V[-1, :4]
            X_manual = X_manual[:3] / X_manual[3]
            
            logger.info(f"Punto 3D manual: {X_manual}")
            logger.info(f"Punto 3D SVD:    {points_3d_svd[0]}")
            
            # Reproyectar manualmente
            for cam_id, pt_2d_orig in first_point_2d.items():
                camera = cameras[cam_id]
                pt_reproj = camera.project_points(X_manual.reshape(1, 3))[0]
                error = np.linalg.norm(pt_reproj - pt_2d_orig)
                logger.info(f"{cam_id}: Original {pt_2d_orig} -> Reproj {pt_reproj} -> Error {error:.2f}px")
        logger.info("=== FIN VERIFICACIÓN ===")
        
        
        # 8. Guardar resultados
        # Formato: {frame_id}_{chunk_id}_method.npy (ej: 3_2_svd.npy)
        
        # Guardar SVD
        svd_file = output_dir / f"{frame_id}_{chunk_id}_svd.npy"
        np.save(svd_file, points_3d_svd)
        
        # Guardar Bundle Adjustment
        ba_file = output_dir / f"{frame_id}_{chunk_id}_ba.npy"
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
        print(f"Estructura: patient{patient_id}/session{session_id}/")
        print(f"Formato archivos: frame_{{frame_id}}_{{chunk_id}}_{{method}}.npy")
    
    print("\n=== FIN PRUEBA ===")


if __name__ == "__main__":
    test_reconstruction_system()

import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import time

# Asegurar que config esté en el path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from .camera import Camera
from .triangulation_svd import triangulate_frame_svd
from .calculate_extrinsics import estimate_extrinsics
from .bundle_adjustment import bundle_adjustment

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Constantes
CONFIDENCE_THRESHOLD = 0.5


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
        coords_file = base_path / camera_id / "coordinates" / f"{frame_id}_{chunk_id}.npy"
        confs_file = base_path / camera_id / "confidence" / f"{frame_id}_{chunk_id}.npy"

        # Verificar que ambos archivos existen
        if coords_file.exists() and confs_file.exists():
            try:
                # Cargar coordenadas y confianzas
                coords = np.load(coords_file)
                confs = np.load(confs_file)
                frame_data[camera_id] = (coords, confs)
            except Exception as e:
                logger.warning(f"Error cargando {camera_id}: {e}")
                continue
        else:
            logger.warning(f"Archivos faltantes para {camera_id} frame {frame_id} chunk {chunk_id}")
    
    logger.debug(f"Cargados keypoints para {len(frame_data)} cámaras, frame {frame_id}, chunk {chunk_id}")
    return frame_data


def create_cameras_from_config() -> Dict[str, Camera]:
    """Crea objetos Camera usando la configuración de intrínsecos"""
    cameras = {}
    for cam_id in ["camera0", "camera1", "camera2"]:
        # Crear cámara base
        cam = Camera.create(cam_id)
        # Los extrínsecos se estimarán posteriormente
        cameras[cam_id] = cam
    return cameras


def get_all_available_frames(patient_id: str, session_id: str) -> Dict[int, List[int]]:
    """
    Obtiene todos los frames disponibles organizados por chunk
    Devuelve un diccionario {chunk_id: [lista_de_frame_ids]}
    Similar a la lógica de ensemble_processor
    """
    base_path = _ROOT / "data" / "processed" / "2D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    
    if not base_path.exists():
        logger.warning(f"No se encontró el directorio de sesión: {base_path}")
        return {}
    
    # Buscar frames comunes en todas las cámaras
    all_frames_by_chunk = {}
    
    for camera_id in ["camera0", "camera1", "camera2"]:
        camera_coords_dir = base_path / camera_id / "coordinates"
        if not camera_coords_dir.exists():
            continue
            
        # Buscar todos los archivos .npy en esta cámara
        frame_files = list(camera_coords_dir.glob("*.npy"))
        
        for frame_file in frame_files:
            try:
                # Extraer frame_id y chunk_id del nombre del archivo: frame_chunk.npy
                parts = frame_file.stem.split('_')
                if len(parts) == 2:
                    frame_id = int(parts[0])
                    chunk_id = int(parts[1])
                    
                    # Verificar que también existe el archivo de confianza
                    confidence_file = base_path / camera_id / "confidence" / frame_file.name
                    if confidence_file.exists():
                        if chunk_id not in all_frames_by_chunk:
                            all_frames_by_chunk[chunk_id] = set()
                        all_frames_by_chunk[chunk_id].add(frame_id)
            except (ValueError, IndexError) as e:
                logger.debug(f"Error procesando archivo {frame_file}: {e}")
                continue
    
    # Encontrar frames comunes entre todas las cámaras para cada chunk
    common_frames_by_chunk = {}
    for chunk_id, frame_set in all_frames_by_chunk.items():
        # Verificar que el frame existe en todas las cámaras
        common_frames = frame_set.copy()
        
        for camera_id in ["camera0", "camera1", "camera2"]:
            camera_coords_dir = base_path / camera_id / "coordinates"
            camera_conf_dir = base_path / camera_id / "confidence"
            
            if camera_coords_dir.exists() and camera_conf_dir.exists():
                camera_frames = set()
                for frame_id in frame_set:
                    coords_file = camera_coords_dir / f"{frame_id}_{chunk_id}.npy"
                    conf_file = camera_conf_dir / f"{frame_id}_{chunk_id}.npy"
                    if coords_file.exists() and conf_file.exists():
                        camera_frames.add(frame_id)
                
                common_frames &= camera_frames
            else:
                common_frames = set()  # Si una cámara no tiene directorio, no hay frames comunes
                break
        
        if common_frames:
            common_frames_by_chunk[chunk_id] = sorted(list(common_frames))
    
    # Log de resumen
    total_frames = sum(len(frames) for frames in common_frames_by_chunk.values())
    logger.info(f"Frames disponibles para reconstrucción: {total_frames} frames en {len(common_frames_by_chunk)} chunks")
    for chunk_id in sorted(common_frames_by_chunk.keys()):
        logger.info(f"  Chunk {chunk_id}: {len(common_frames_by_chunk[chunk_id])} frames")
    
    return common_frames_by_chunk

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
    
    logger.debug(f"Factor de escala calculado: {scale_factor:.4f} (altura: {person_height_cm}cm)")
    return scale_factor


def average_rotation_matrices(rotation_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Calcula el promedio de matrices de rotación preservando las propiedades de rotación.
    Usa SVD para encontrar la matriz de rotación más cercana al promedio.
    """
    if len(rotation_matrices) == 1:
        return rotation_matrices[0]
    
    # Promedio simple como aproximación inicial
    avg_matrix = np.mean(np.array(rotation_matrices), axis=0)
    
    # Usar SVD para encontrar la matriz de rotación más cercana
    U, _, Vt = np.linalg.svd(avg_matrix)
    
    # La matriz de rotación más cercana es U * Vt
    R_averaged = U @ Vt
    
    # Asegurar determinante positivo (rotación, no reflexión)
    if np.linalg.det(R_averaged) < 0:
        U[:, -1] *= -1
        R_averaged = U @ Vt
    
    return R_averaged


def calculate_session_extrinsics(patient_id: str, session_id: str, available_frames: Dict[int, List[int]], max_chunk: int) -> Optional[Dict[str, Camera]]:
    """
    Calcula los extrínsecos de las cámaras usando múltiples frames para mayor robustez.
    Usa el frame 15 de cada chunk (desde 0 hasta max_chunk) y promedia las matrices resultantes.
    Los extrínsecos son consistentes durante toda la sesión ya que las cámaras no se mueven.
    """
    logger.info(f"Calculando extrínsecos robustos usando frame 15 de chunks 0-{max_chunk}...")
    
    # Recopilar matrices extrínsecas de múltiples chunks
    collected_extrinsics = {}  # {camera_id: [lista_de_matrices_R, lista_de_vectores_t]}
    successful_calculations = 0
    target_frame_id = 15
    
    # Inicializar estructura para cada cámara
    for cam_id in ["camera0", "camera1", "camera2"]:
        collected_extrinsics[cam_id] = {'R_matrices': [], 't_vectors': []}
    
    # Intentar calcular extrínsecos para cada chunk
    for chunk_id in range(max_chunk + 1):
        logger.debug(f"Intentando calcular extrínsecos para chunk {chunk_id}...")
        
        # Verificar si el chunk y frame existen
        if chunk_id not in available_frames:
            logger.debug(f"Chunk {chunk_id} no disponible, saltando...")
            continue
            
        if target_frame_id not in available_frames[chunk_id]:
            logger.debug(f"Frame {target_frame_id} no disponible en chunk {chunk_id}, saltando...")
            continue
        
        try:
            # Cargar keypoints del frame específico
            frame_keypoints = load_frame_keypoints(patient_id, session_id, chunk_id, target_frame_id)
            
            if len(frame_keypoints) < 2:
                logger.debug(f"Chunk {chunk_id}: insuficientes cámaras con datos")
                continue
            
            # Crear cámaras y estimar extrínsecos para este chunk
            cameras = create_cameras_from_config()
            chunk_cameras = estimate_extrinsics(cameras, frame_keypoints, CONFIDENCE_THRESHOLD)
            
            # Almacenar matrices resultantes
            for cam_id, camera in chunk_cameras.items():
                if hasattr(camera, 'R') and hasattr(camera, 't'):
                    collected_extrinsics[cam_id]['R_matrices'].append(camera.R.copy())
                    collected_extrinsics[cam_id]['t_vectors'].append(camera.t.flatten().copy())
            
            successful_calculations += 1
            logger.debug(f"Chunk {chunk_id}: extrínsecos calculados exitosamente")
            
        except Exception as e:
            logger.debug(f"Chunk {chunk_id}: error calculando extrínsecos - {e}")
            continue
    
    if successful_calculations == 0:
        logger.error("No se pudieron calcular extrínsecos para ningún chunk")
        return None
    
    logger.info(f"Extrínsecos calculados exitosamente para {successful_calculations} chunks")
    
    # Calcular matrices promedio para cada cámara
    try:
        averaged_cameras = create_cameras_from_config()
        
        for cam_id in ["camera0", "camera1", "camera2"]:
            R_matrices = collected_extrinsics[cam_id]['R_matrices']
            t_vectors = collected_extrinsics[cam_id]['t_vectors']
            
            if len(R_matrices) == 0:
                logger.warning(f"No se encontraron matrices para {cam_id}, usando identidad")
                continue
            
            # Promedio de matrices de rotación (usando función especializada)
            avg_R = average_rotation_matrices(R_matrices)
            
            # Promedio de vectores de traslación
            avg_t = np.mean(np.array(t_vectors), axis=0).reshape(3, 1)
            
            # Asignar matrices promediadas
            averaged_cameras[cam_id].R = avg_R
            averaged_cameras[cam_id].t = avg_t
            
            logger.debug(f"{cam_id}: promediado {len(R_matrices)} matrices R y vectores t")
        
        logger.info("Matrices extrínsecas promediadas calculadas exitosamente")
        return averaged_cameras
        
    except Exception as e:
        logger.error(f"Error calculando matrices promedio: {e}")
        return None


def reconstruct_frame_with_extrinsics(cameras: Dict[str, Camera], frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                     person_height_cm: float) -> Optional[np.ndarray]:
    """
    Reconstruye un frame usando cámaras con extrínsecos pre-calculados.
    Elimina la duplicación de código para el cálculo de extrínsecos.
    """
    try:
        # Triangular inicial con SVD
        points_3d_svd = triangulate_frame_svd(cameras, frame_keypoints, CONFIDENCE_THRESHOLD)

        # Aplicar Bundle Adjustment
        points_3d_full_ba, _ = bundle_adjustment(
            points_3d_svd, cameras, frame_keypoints, 
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Aplicar escalado por altura
        scale_factor = calculate_scale_factor_from_height(points_3d_full_ba, person_height_cm)
        if scale_factor:
            points_3d_full_ba = points_3d_full_ba * scale_factor
            
        return points_3d_full_ba
            
    except Exception as e:
        logger.debug(f"Error en reconstrucción: {e}")
        return None


def save_3d_reconstruction(points_3d: np.ndarray, patient_id: str, session_id: str, frame_id: int, chunk_id: int):
    """Guarda la reconstrucción 3D en formato .npy con el nombre específico del frame y chunk"""
    
    # Crear directorio de salida
    output_dir = _ROOT / "data" / "processed" / "3D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar puntos 3D con el mismo formato que los 2D: frame_chunk.npy
    output_file = output_dir / f"{frame_id}_{chunk_id}.npy"
    np.save(output_file, points_3d)
    
    logger.debug(f"Reconstrucción 3D guardada: {output_file}")
    return output_file


def process_chunk_parallel(chunk_data: Tuple[int, List[int], str, str, Dict[str, Camera], float]) -> Dict[str, any]:
    """
    Procesa un chunk completo de frames en paralelo.
    
    Args:
        chunk_data: (chunk_id, frame_list, patient_id, session_id, session_cameras, person_height_cm)
    
    Returns:
        Dict con estadísticas del chunk procesado
    """
    chunk_id, frame_list, patient_id, session_id, session_cameras, person_height_cm = chunk_data
    
    # Configurar logging para este proceso
    logger = logging.getLogger(f"chunk_{chunk_id}")
    
    chunk_successes = 0
    chunk_start_time = time.time()
    
    logger.info(f"Iniciando procesamiento del chunk {chunk_id}: {len(frame_list)} frames")
    
    for frame_id in frame_list:
        try:
            # Cargar keypoints del frame
            frame_keypoints = load_frame_keypoints(patient_id, session_id, chunk_id, frame_id)
            
            if len(frame_keypoints) < 2:
                logger.debug(f"Frame {frame_id}: insuficientes cámaras con datos")
                continue
            
            # Reconstruir usando extrínsecos pre-calculados
            points_3d_reconstructed = reconstruct_frame_with_extrinsics(
                session_cameras, frame_keypoints, person_height_cm
            )
            
            if points_3d_reconstructed is not None:
                # Guardar resultado
                save_3d_reconstruction(points_3d_reconstructed, patient_id, session_id, frame_id, chunk_id)
                chunk_successes += 1
            else:
                logger.debug(f"Frame {frame_id}: falló la reconstrucción")

        except Exception as e:
            logger.debug(f"Error procesando frame {frame_id}, chunk {chunk_id}: {e}")
    
    chunk_time = time.time() - chunk_start_time
    logger.info(f"Chunk {chunk_id} completado en {chunk_time:.2f}s: {chunk_successes}/{len(frame_list)} reconstrucciones exitosas")
    
    return {
        'chunk_id': chunk_id,
        'total_frames': len(frame_list),
        'successful_frames': chunk_successes,
        'processing_time': chunk_time
    }


def start_3d_reconstruction(patient_id: str, session_id: str, max_chunk: int, person_height_cm: float = 190.0, use_parallel: bool = True):
    """
    Función principal optimizada que ejecuta reconstrucciones 3D para todos los frames.
    Calcula los extrínsecos una vez por sesión y los reutiliza para todos los frames.
    Opcionalmente puede procesar chunks en paralelo para mayor velocidad.
    """
    start_time = time.time()
    logger.info(f"Iniciando reconstrucciones 3D para patient{patient_id}/session{session_id}")
    logger.info(f"Max chunk: {max_chunk}, Altura persona: {person_height_cm} cm")
    logger.info(f"Procesamiento paralelo: {'ACTIVADO' if use_parallel else 'DESACTIVADO'}")
    
    # Obtener todos los frames disponibles
    available_frames = get_all_available_frames(patient_id, session_id)
    
    if not available_frames:
        logger.warning("No se encontraron frames disponibles para reconstrucción")
        return {}
    
    # Calcular extrínsecos para toda la sesión usando múltiples chunks
    session_cameras = calculate_session_extrinsics(patient_id, session_id, available_frames, max_chunk)
    
    if session_cameras is None:
        logger.error("No se pudieron calcular los extrínsecos de la sesión")
        return {}
    
    # Contar total de frames
    total_frames = sum(len(frames) for frames in available_frames.values())
    logger.info(f"Procesando {total_frames} frames en {len(available_frames)} chunks con extrínsecos fijos")
    
    successful_reconstructions = 0
    processing_results = []
    
    if use_parallel:
        # PROCESAMIENTO PARALELO
        num_processes = min(cpu_count(), len(available_frames))  # No usar más procesos que chunks
        logger.info(f"Usando {num_processes} procesos paralelos para {len(available_frames)} chunks")
        
        # Preparar datos para cada chunk
        chunk_data_list = []
        for chunk_id in sorted(available_frames.keys()):
            frame_list = available_frames[chunk_id]
            chunk_data = (chunk_id, frame_list, patient_id, session_id, session_cameras, person_height_cm)
            chunk_data_list.append(chunk_data)
        
        # Procesar chunks en paralelo
        with Pool(processes=num_processes) as pool:
            processing_results = pool.map(process_chunk_parallel, chunk_data_list)
        
        # Consolidar resultados
        successful_reconstructions = sum(result['successful_frames'] for result in processing_results)
        
    else:
        # PROCESAMIENTO SECUENCIAL (como antes)
        for chunk_id in sorted(available_frames.keys()):
            frame_list = available_frames[chunk_id]
            logger.info(f"Procesando chunk {chunk_id}: {len(frame_list)} frames")
            
            chunk_data = (chunk_id, frame_list, patient_id, session_id, session_cameras, person_height_cm)
            result = process_chunk_parallel(chunk_data)  # Reutilizamos la misma función
            processing_results.append(result)
            successful_reconstructions += result['successful_frames']
    
    # Estadísticas finales
    total_time = time.time() - start_time
    success_rate = (successful_reconstructions / total_frames) * 100 if total_frames > 0 else 0
    
    logger.info(f"RECONSTRUCCIONES 3D COMPLETADAS en {total_time:.2f}s:")
    logger.info(f"Total de frames procesados: {total_frames}")
    logger.info(f"Reconstrucciones exitosas: {successful_reconstructions} ({success_rate:.1f}%)")
    
    if use_parallel:
        # Mostrar tiempos por chunk para análisis
        for result in processing_results:
            chunk_id = result['chunk_id']
            chunk_time = result['processing_time']
            chunk_success = result['successful_frames']
            chunk_total = result['total_frames']
            logger.info(f"  Chunk {chunk_id}: {chunk_success}/{chunk_total} frames en {chunk_time:.2f}s")
    
    return {
        'total_frames': total_frames,
        'successful_reconstructions': successful_reconstructions,
        'success_rate': success_rate,
        'total_time': total_time,
        'parallel_processing': use_parallel,
        'chunk_results': processing_results
    }


if __name__ == "__main__":
    patient_id = 57
    session_id = 57
    max_chunk = 7
    person_height_cm = 190.0
    
    # Puedes cambiar use_parallel=False para procesamiento secuencial
    use_parallel = True
    
    print(f"Iniciando reconstrucción 3D con procesamiento {'PARALELO' if use_parallel else 'SECUENCIAL'}")
    results = start_3d_reconstruction(patient_id, session_id, max_chunk, person_height_cm, use_parallel)
    
    if results:
        print(f"\n{'='*60}")
        print("RESULTADOS FINALES:")
        print(f"Tiempo total: {results.get('total_time', 0):.2f} segundos")
        print(f"Frames procesados: {results['successful_reconstructions']}/{results['total_frames']}")
        print(f"Tasa de éxito: {results['success_rate']:.1f}%")
        print(f"Procesamiento paralelo: {'SÍ' if results['parallel_processing'] else 'NO'}")
        print(f"{'='*60}")
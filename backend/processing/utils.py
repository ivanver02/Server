"""
Funciones auxiliares para el manejo de archivos de keypoints
Utilidades para lectura, escritura y validación de datos del pipeline
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json

logger = logging.getLogger(__name__)


def save_keypoints_2d(keypoints: np.ndarray, 
                     scores: np.ndarray,
                     camera_id: int,
                     frame_number: int,
                     detector_name: str,
                     patient_id: str,
                     session_id: str,
                     chunk_number: int,
                     base_data_dir: Path) -> Tuple[Path, Path]:
    """
    Guardar keypoints 2D y scores en archivos .npy
    
    Args:
        keypoints: Array de keypoints (N, 2)
        scores: Array de confianza (N,)
        camera_id: ID de la cámara
        frame_number: Número del frame
        detector_name: Nombre del detector
        patient_id: ID del paciente
        session_id: ID de la sesión
        chunk_number: Número del chunk
        base_data_dir: Directorio base de datos
        
    Returns:
        Tuple con paths de keypoints y confidence files
    """
    try:
        # Calcular global_frame
        global_frame = chunk_number * 1000 + frame_number
        
        # Directorio de la cámara
        camera_dir = base_data_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        camera_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar keypoints
        keypoints_file = camera_dir / f"{global_frame}_{detector_name}.npy"
        np.save(keypoints_file, keypoints)
        
        # Guardar confianzas
        confidence_file = camera_dir / f"{global_frame}_{detector_name}_confidence.npy"
        np.save(confidence_file, scores)
        
        return keypoints_file, confidence_file
        
    except Exception as e:
        logger.error(f"Error guardando keypoints 2D: {e}")
        raise


def load_keypoints_2d(camera_id: int,
                     frame_number: int, 
                     detector_name: str,
                     patient_id: str,
                     session_id: str,
                     chunk_number: int,
                     base_data_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Cargar keypoints 2D y scores desde archivos .npy
    
    Args:
        camera_id: ID de la cámara
        frame_number: Número del frame
        detector_name: Nombre del detector
        patient_id: ID del paciente
        session_id: ID de la sesión
        chunk_number: Número del chunk
        base_data_dir: Directorio base de datos
        
    Returns:
        Tuple (keypoints, scores) o (None, None) si no existe
    """
    try:
        # Calcular global_frame
        global_frame = chunk_number * 1000 + frame_number
        
        # Directorio de la cámara
        camera_dir = base_data_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        
        # Archivos
        keypoints_file = camera_dir / f"{global_frame}_{detector_name}.npy"
        confidence_file = camera_dir / f"{global_frame}_{detector_name}_confidence.npy"
        
        if keypoints_file.exists() and confidence_file.exists():
            keypoints = np.load(keypoints_file)
            scores = np.load(confidence_file)
            return keypoints, scores
        else:
            return None, None
            
    except Exception as e:
        logger.error(f"Error cargando keypoints 2D: {e}")
        return None, None


def get_available_detections(patient_id: str,
                           session_id: str,
                           chunk_number: int,
                           base_data_dir: Path) -> Dict[int, Dict[int, List[str]]]:
    """
    Obtener detecciones disponibles para un chunk
    
    Args:
        patient_id: ID del paciente
        session_id: ID de la sesión
        chunk_number: Número del chunk
        base_data_dir: Directorio base de datos
        
    Returns:
        Dict {camera_id: {frame_number: [detector_names]}}
    """
    try:
        session_dir = base_data_dir / f"patient{patient_id}" / f"session{session_id}"
        detections = {}
        
        # Buscar en todas las cámaras
        for camera_dir in session_dir.glob("camera*"):
            try:
                camera_id = int(camera_dir.name.replace("camera", ""))
                detections[camera_id] = {}
                
                # Buscar archivos de keypoints para este chunk
                chunk_prefix = f"{chunk_number * 1000}"
                
                for keypoints_file in camera_dir.glob(f"{chunk_prefix}*_*.npy"):
                    # Excluir archivos de confidence
                    if "_confidence.npy" in str(keypoints_file):
                        continue
                    
                    # Extraer frame number y detector name
                    filename = keypoints_file.stem  # sin .npy
                    parts = filename.split("_")
                    
                    if len(parts) >= 2:
                        global_frame = int(parts[0])
                        frame_number = global_frame - (chunk_number * 1000)
                        detector_name = "_".join(parts[1:])
                        
                        if frame_number not in detections[camera_id]:
                            detections[camera_id][frame_number] = []
                        
                        detections[camera_id][frame_number].append(detector_name)
                        
            except (ValueError, IndexError) as e:
                logger.warning(f"Error procesando directorio {camera_dir}: {e}")
                continue
        
        return detections
        
    except Exception as e:
        logger.error(f"Error obteniendo detecciones disponibles: {e}")
        return {}


def validate_keypoints_integrity(patient_id: str,
                               session_id: str,
                               chunk_number: int,
                               expected_cameras: List[int],
                               expected_detectors: List[str],
                               base_data_dir: Path) -> Dict[str, Any]:
    """
    Validar integridad de los keypoints guardados
    
    Args:
        patient_id: ID del paciente
        session_id: ID de la sesión
        chunk_number: Número del chunk
        expected_cameras: Lista de IDs de cámaras esperadas
        expected_detectors: Lista de detectores esperados
        base_data_dir: Directorio base de datos
        
    Returns:
        Reporte de integridad
    """
    try:
        detections = get_available_detections(patient_id, session_id, chunk_number, base_data_dir)
        
        report = {
            'chunk_number': chunk_number,
            'expected_cameras': expected_cameras,
            'expected_detectors': expected_detectors,
            'cameras_found': list(detections.keys()),
            'missing_cameras': [],
            'frames_per_camera': {},
            'detectors_per_frame': {},
            'incomplete_frames': [],
            'total_frames': 0,
            'integrity_score': 0.0
        }
        
        # Verificar cámaras faltantes
        report['missing_cameras'] = [cam for cam in expected_cameras if cam not in detections]
        
        # Analizar frames por cámara
        all_frame_numbers = set()
        for camera_id, camera_detections in detections.items():
            report['frames_per_camera'][camera_id] = len(camera_detections)
            all_frame_numbers.update(camera_detections.keys())
        
        report['total_frames'] = len(all_frame_numbers)
        
        # Verificar completitud por frame
        for frame_num in sorted(all_frame_numbers):
            frame_completeness = {}
            incomplete = False
            
            for camera_id in expected_cameras:
                if camera_id in detections and frame_num in detections[camera_id]:
                    frame_detectors = detections[camera_id][frame_num]
                    frame_completeness[camera_id] = frame_detectors
                    
                    # Verificar si tiene todos los detectores esperados
                    missing_detectors = [d for d in expected_detectors if d not in frame_detectors]
                    if missing_detectors:
                        incomplete = True
                else:
                    frame_completeness[camera_id] = []
                    incomplete = True
            
            report['detectors_per_frame'][frame_num] = frame_completeness
            if incomplete:
                report['incomplete_frames'].append(frame_num)
        
        # Calcular score de integridad
        if report['total_frames'] > 0:
            complete_frames = report['total_frames'] - len(report['incomplete_frames'])
            camera_completeness = len(expected_cameras) - len(report['missing_cameras'])
            
            report['integrity_score'] = (complete_frames / report['total_frames']) * (camera_completeness / len(expected_cameras))
        
        return report
        
    except Exception as e:
        logger.error(f"Error validando integridad de keypoints: {e}")
        return {'error': str(e)}


def cleanup_chunk_files(patient_id: str,
                       session_id: str,
                       chunk_number: int,
                       base_data_dir: Path,
                       keep_metadata: bool = True) -> bool:
    """
    Limpiar archivos de un chunk específico
    
    Args:
        patient_id: ID del paciente
        session_id: ID de la sesión
        chunk_number: Número del chunk
        base_data_dir: Directorio base de datos
        keep_metadata: Si mantener archivos de metadatos
        
    Returns:
        True si se limpió correctamente
    """
    try:
        session_dir = base_data_dir / f"patient{patient_id}" / f"session{session_id}"
        chunk_prefix = f"{chunk_number * 1000}"
        files_deleted = 0
        
        for camera_dir in session_dir.glob("camera*"):
            # Eliminar archivos de keypoints
            for file_path in camera_dir.glob(f"{chunk_prefix}*_*.npy"):
                file_path.unlink()
                files_deleted += 1
            
            # Eliminar metadatos si no se requiere mantenerlos
            if not keep_metadata:
                for file_path in camera_dir.glob(f"{chunk_prefix}*_metadata.json"):
                    file_path.unlink()
                    files_deleted += 1
        
        logger.info(f"Eliminados {files_deleted} archivos del chunk {chunk_number}")
        return True
        
    except Exception as e:
        logger.error(f"Error limpiando archivos del chunk: {e}")
        return False


def get_chunk_statistics(patient_id: str,
                        session_id: str,
                        base_data_dir: Path) -> Dict[str, Any]:
    """
    Obtener estadísticas de todos los chunks de una sesión
    
    Args:
        patient_id: ID del paciente
        session_id: ID de la sesión
        base_data_dir: Directorio base de datos
        
    Returns:
        Estadísticas detalladas
    """
    try:
        session_dir = base_data_dir / f"patient{patient_id}" / f"session{session_id}"
        
        stats = {
            'patient_id': patient_id,
            'session_id': session_id,
            'cameras_detected': [],
            'chunks_available': set(),
            'total_frames': 0,
            'detectors_used': set(),
            'file_count': {
                'keypoints': 0,
                'confidence': 0,
                'metadata': 0
            },
            'size_mb': 0.0
        }
        
        for camera_dir in session_dir.glob("camera*"):
            try:
                camera_id = int(camera_dir.name.replace("camera", ""))
                stats['cameras_detected'].append(camera_id)
                
                for file_path in camera_dir.iterdir():
                    if file_path.is_file():
                        # Analizar nombre del archivo
                        if file_path.suffix == '.npy':
                            parts = file_path.stem.split('_')
                            if len(parts) >= 2:
                                global_frame = int(parts[0])
                                chunk_num = global_frame // 1000
                                stats['chunks_available'].add(chunk_num)
                                
                                if '_confidence' in file_path.stem:
                                    stats['file_count']['confidence'] += 1
                                else:
                                    stats['file_count']['keypoints'] += 1
                                    detector_name = "_".join(parts[1:])
                                    stats['detectors_used'].add(detector_name)
                        
                        elif file_path.suffix == '.json':
                            stats['file_count']['metadata'] += 1
                        
                        # Sumar tamaño
                        stats['size_mb'] += file_path.stat().st_size / (1024 * 1024)
                        
            except (ValueError, OSError) as e:
                logger.warning(f"Error procesando {camera_dir}: {e}")
                continue
        
        # Convertir sets a listas ordenadas
        stats['cameras_detected'].sort()
        stats['chunks_available'] = sorted(list(stats['chunks_available']))
        stats['detectors_used'] = sorted(list(stats['detectors_used']))
        
        # Estimar frames totales
        stats['total_frames'] = stats['file_count']['keypoints']
        
        return stats
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return {'error': str(e)}

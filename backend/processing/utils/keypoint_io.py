"""
Funciones auxiliares para manejo de archivos de keypoints 2D
Estructura organizada: keypoints/ y confidence/ por separado
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


def save_keypoints_2d_frame(keypoints: np.ndarray, 
                           confidence: np.ndarray,
                           detector_name: str,
                           camera_id: int,
                           global_frame: int,
                           patient_id: str,
                           session_id: str) -> Tuple[str, str]:
    """
    Guardar keypoints 2D y confianza de un frame en estructura organizada
    
    Estructura:
    data/patient{}/session{}/camera{}/
    ├── keypoints/
    │   └── {global_frame}_{detector_name}.npy
    └── confidence/
        └── {global_frame}_{detector_name}.npy
    
    Args:
        keypoints: Array de coordenadas (N, 2)
        confidence: Array de confianza (N,)
        detector_name: Nombre del detector
        camera_id: ID de la cámara
        global_frame: Número de frame global
        patient_id: ID del paciente
        session_id: ID de la sesión
        
    Returns:
        Tuple con las rutas de los archivos guardados (keypoints_path, confidence_path)
    """
    try:
        from config import data_config
        
        # Directorio base de la cámara
        camera_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        
        # Crear subdirectorios
        keypoints_dir = camera_dir / "keypoints"
        confidence_dir = camera_dir / "confidence"
        keypoints_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombres de archivos
        base_filename = f"{global_frame}_{detector_name}.npy"
        keypoints_file = keypoints_dir / base_filename
        confidence_file = confidence_dir / base_filename
        
        # Guardar arrays
        np.save(keypoints_file, keypoints)
        np.save(confidence_file, confidence)
        
        return str(keypoints_file), str(confidence_file)
        
    except Exception as e:
        logger.error(f"Error guardando keypoints 2D frame {global_frame}: {e}")
        raise


def load_keypoints_2d_frame(detector_name: str,
                           camera_id: int,
                           global_frame: int,
                           patient_id: str,
                           session_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Cargar keypoints 2D y confianza de un frame específico
    
    Args:
        detector_name: Nombre del detector
        camera_id: ID de la cámara
        global_frame: Número de frame global
        patient_id: ID del paciente
        session_id: ID de la sesión
        
    Returns:
        Tuple (keypoints, confidence) o (None, None) si no existe
    """
    try:
        from config import data_config
        
        # Rutas de archivos
        camera_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        base_filename = f"{global_frame}_{detector_name}.npy"
        
        keypoints_file = camera_dir / "keypoints" / base_filename
        confidence_file = camera_dir / "confidence" / base_filename
        
        # Verificar que ambos archivos existan
        if not (keypoints_file.exists() and confidence_file.exists()):
            return None, None
        
        # Cargar arrays
        keypoints = np.load(keypoints_file)
        confidence = np.load(confidence_file)
        
        return keypoints, confidence
        
    except Exception as e:
        logger.error(f"Error cargando keypoints 2D frame {global_frame}: {e}")
        return None, None


def get_available_frames(camera_id: int,
                        patient_id: str,
                        session_id: str,
                        detector_name: Optional[str] = None) -> List[int]:
    """
    Obtener lista de frames disponibles para una cámara
    
    Args:
        camera_id: ID de la cámara
        patient_id: ID del paciente
        session_id: ID de la sesión
        detector_name: Filtrar por detector específico (opcional)
        
    Returns:
        Lista de números de frame disponibles
    """
    try:
        from config import data_config
        
        camera_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        keypoints_dir = camera_dir / "keypoints"
        
        if not keypoints_dir.exists():
            return []
        
        frames = set()
        
        # Buscar archivos de keypoints
        for file in keypoints_dir.glob("*.npy"):
            try:
                # Formato: {global_frame}_{detector_name}.npy
                parts = file.stem.split('_', 1)
                if len(parts) == 2:
                    frame_num = int(parts[0])
                    file_detector = parts[1]
                    
                    # Filtrar por detector si se especifica
                    if detector_name is None or file_detector == detector_name:
                        frames.add(frame_num)
                        
            except ValueError:
                continue
        
        return sorted(list(frames))
        
    except Exception as e:
        logger.error(f"Error obteniendo frames disponibles: {e}")
        return []


def get_available_detectors(camera_id: int,
                           patient_id: str,
                           session_id: str,
                           global_frame: Optional[int] = None) -> List[str]:
    """
    Obtener lista de detectores disponibles para una cámara
    
    Args:
        camera_id: ID de la cámara
        patient_id: ID del paciente
        session_id: ID de la sesión
        global_frame: Filtrar por frame específico (opcional)
        
    Returns:
        Lista de nombres de detectores disponibles
    """
    try:
        from config import data_config
        
        camera_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        keypoints_dir = camera_dir / "keypoints"
        
        if not keypoints_dir.exists():
            return []
        
        detectors = set()
        
        # Buscar archivos de keypoints
        for file in keypoints_dir.glob("*.npy"):
            try:
                # Formato: {global_frame}_{detector_name}.npy
                parts = file.stem.split('_', 1)
                if len(parts) == 2:
                    file_frame = int(parts[0])
                    detector_name = parts[1]
                    
                    # Filtrar por frame si se especifica
                    if global_frame is None or file_frame == global_frame:
                        detectors.add(detector_name)
                        
            except ValueError:
                continue
        
        return sorted(list(detectors))
        
    except Exception as e:
        logger.error(f"Error obteniendo detectores disponibles: {e}")
        return []


def save_frame_metadata(metadata: Dict,
                       camera_id: int,
                       global_frame: int,
                       patient_id: str,
                       session_id: str) -> str:
    """
    Guardar metadata de un frame (opcional, para debugging)
    
    Args:
        metadata: Diccionario con metadata del frame
        camera_id: ID de la cámara
        global_frame: Número de frame global
        patient_id: ID del paciente
        session_id: ID de la sesión
        
    Returns:
        Ruta del archivo de metadata guardado
    """
    try:
        from config import data_config
        
        # Directorio de metadata
        camera_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        metadata_dir = camera_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivo de metadata
        metadata_file = metadata_dir / f"{global_frame}_frame.json"
        
        # Guardar metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_file)
        
    except Exception as e:
        logger.error(f"Error guardando metadata frame {global_frame}: {e}")
        raise


def get_session_summary(patient_id: str, session_id: str) -> Dict:
    """
    Obtener resumen de una sesión completa
    
    Args:
        patient_id: ID del paciente
        session_id: ID de la sesión
        
    Returns:
        Diccionario con resumen de la sesión
    """
    try:
        from config import data_config
        
        session_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}"
        
        if not session_dir.exists():
            return {'error': 'Session not found'}
        
        summary = {
            'patient_id': patient_id,
            'session_id': session_id,
            'cameras': {},
            'total_frames': 0,
            'available_detectors': set()
        }
        
        # Analizar cada cámara
        for camera_dir in session_dir.glob("camera*"):
            try:
                camera_id = int(camera_dir.name.replace('camera', ''))
                
                frames = get_available_frames(camera_id, patient_id, session_id)
                detectors = get_available_detectors(camera_id, patient_id, session_id)
                
                summary['cameras'][camera_id] = {
                    'total_frames': len(frames),
                    'frame_range': (min(frames), max(frames)) if frames else (0, 0),
                    'detectors': detectors
                }
                
                summary['total_frames'] += len(frames)
                summary['available_detectors'].update(detectors)
                
            except ValueError:
                continue
        
        summary['available_detectors'] = sorted(list(summary['available_detectors']))
        
        return summary
        
    except Exception as e:
        logger.error(f"Error obteniendo resumen de sesión: {e}")
        return {'error': str(e)}

"""
Procesador de ensemble generalizable para combinar múltiples detectores de pose
"""
import numpy as np
import threading
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Importar las clases de detectores
from backend.processing.detectors.vitpose import VitPoseDetector
from backend.processing.detectors.csp import CSPDetector
from backend.processing.detectors.hrnet import HRNetDetector

# Importar configuración de ensemble
from config.settings import ensemble_config

logger = logging.getLogger(__name__)

class EnsembleProcessor:
    """
    Procesador de ensemble que combina detectores cuando todas las cámaras han terminado
    """
    
    def __init__(self, base_data_dir: Path):
        self.base_data_dir = Path(base_data_dir)
        
        # Inicializar instancias de detectores para acceder a sus configuraciones
        self.detector_instances = {
            'vitpose': VitPoseDetector(),
            'csp': CSPDetector(),
            'hrnet': HRNetDetector()
        }
        
        # Generar lista de keypoints finales basada en ponderaciones no nulas
        self.final_keypoint_names = self._generate_final_keypoint_names()
        
        # Registro de sesiones activas: {patient_id: {session_id: {'max_chunk': int, 'cameras_count': int}}}
        self.active_sessions = {}
        
        logger.info(f"EnsembleProcessor inicializado con {len(self.final_keypoint_names)} keypoints finales")
        logger.info(f"Keypoints finales: {self.final_keypoint_names}")
        logger.info(f"Configuración ensemble: umbral_confianza={ensemble_config.confidence_threshold}, min_detectores={ensemble_config.min_detectors_required}")
    
    def _generate_final_keypoint_names(self) -> List[str]:
        """
        Generar lista de nombres de keypoints finales basada en ponderaciones no nulas
        """
        final_names = []
        
        # Obtener el número máximo de keypoints entre todos los detectores
        max_keypoints = max(
            len(detector.ensemble_confidence_weights) 
            for detector in self.detector_instances.values()
        )
        
        for kp_idx in range(max_keypoints):
            # Verificar si algún detector tiene ponderación no nula para este keypoint
            has_weight = False
            keypoint_name = None
            
            for detector_name, detector in self.detector_instances.items():
                if (kp_idx < len(detector.ensemble_confidence_weights) and 
                    detector.ensemble_confidence_weights[kp_idx] > 0):
                    has_weight = True
                    if kp_idx < len(detector.keypoints_names):
                        keypoint_name = detector.keypoints_names[kp_idx]
                    break
            
            if has_weight and keypoint_name:
                final_names.append(keypoint_name)
        
        return final_names
    
    def register_session_start(self, patient_id: str, session_id: str, cameras_count: int):
        """Registra el inicio de una sesión para tracking"""
        if patient_id not in self.active_sessions:
            self.active_sessions[patient_id] = {}
        
        self.active_sessions[patient_id][session_id] = {
            'max_chunk': -1,
            'cameras_count': cameras_count,
            'completed_cameras': set()
        }
        
        logger.info(f"Sesión registrada: patient_id={patient_id}, session_id={session_id}, cameras_count={cameras_count}")

    def register_session_end(self, patient_id: str, session_id: str) -> int:
        """Registra el final de una sesión y determina el max_chunk"""
        if patient_id not in self.active_sessions or session_id not in self.active_sessions[patient_id]:
            logger.warning(f"Sesión no encontrada para finalizar: patient_id={patient_id}, session_id={session_id}")
            return -1
        
        # Buscar el chunk máximo para esta sesión en los datos sin procesar
        unprocessed_session_path = self.base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}"
        max_chunk = -1
        
        if unprocessed_session_path.exists():
            # Iterar por todas las carpetas de cámaras
            for camera_folder in unprocessed_session_path.iterdir():
                if camera_folder.is_dir() and camera_folder.name.startswith('camera'):
                    # Buscar archivos de video (chunks) en cada cámara
                    for chunk_file in camera_folder.glob("*.mp4"):
                        try:
                            chunk_num = int(chunk_file.stem)  # El nombre del archivo es el número de chunk
                            max_chunk = max(max_chunk, chunk_num)
                        except (ValueError, TypeError):
                            continue
        
        self.active_sessions[patient_id][session_id]['max_chunk'] = max_chunk
        logger.info(f"Sesión finalizada: patient_id={patient_id}, session_id={session_id}, max_chunk={max_chunk}")
        
        return max_chunk

    def register_chunk_completion(self, patient_id: str, session_id: str, camera_id: str, chunk_id: int):
        """Registra que una cámara completó el procesamiento de un chunk"""
        if patient_id not in self.active_sessions or session_id not in self.active_sessions[patient_id]:
            logger.warning(f"Sesión no encontrada para chunk completion: patient_id={patient_id}, session_id={session_id}")
            return False
            
        session_data = self.active_sessions[patient_id][session_id]
        max_chunk = session_data['max_chunk']
        
        # Solo procesar si es el chunk final
        if max_chunk == -1 or chunk_id != max_chunk:
            return False
            
        session_data['completed_cameras'].add(camera_id)
        logger.info(f"Chunk final completado: patient_id={patient_id}, session_id={session_id}, camera_id={camera_id}, chunk_id={chunk_id}")
        logger.info(f"Cámaras completadas: {len(session_data['completed_cameras'])}/{session_data['cameras_count']}")
        
        # Verificar si todas las cámaras completaron el chunk final
        if len(session_data['completed_cameras']) >= session_data['cameras_count']:
            logger.info(f"¡Todas las cámaras completaron el chunk final! Iniciando ensemble para session_id={session_id}")
            # Ejecutar ensemble en thread separado
            threading.Thread(
                target=self._process_session_ensemble_async,
                args=(patient_id, session_id),
                daemon=True
            ).start()
            return True
            
        return False

    def _process_session_ensemble_async(self, patient_id: str, session_id: str):
        """Procesa el ensemble de forma asíncrona"""
        try:
            self.process_session_ensemble(patient_id, session_id)
            
            # Limpiar sesión después del ensemble
            if patient_id in self.active_sessions and session_id in self.active_sessions[patient_id]:
                del self.active_sessions[patient_id][session_id]
                if not self.active_sessions[patient_id]:
                    del self.active_sessions[patient_id]
                    
        except Exception as e:
            logger.error(f"Error en ensemble asíncrono: {e}")

    def process_session_ensemble(self, patient_id: str, session_id: str):
        """
        Procesar ensemble para toda la sesión cuando todas las cámaras han terminado
        """
        try:
            logger.info(f"🎯 Iniciando ensemble para sesión completa: patient{patient_id}/session{session_id}")
            
            # Obtener lista de cámaras y encontrar el chunk máximo
            unprocessed_session = self.base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}"
            camera_dirs = list(unprocessed_session.glob("camera*"))
            
            if not camera_dirs:
                logger.warning("No se encontraron directorios de cámaras")
                return
            
            # Encontrar el chunk máximo
            max_chunk = self._find_max_chunk(patient_id, session_id)
            if max_chunk < 0:
                logger.warning("No se encontraron chunks para procesar")
                return
            
            total_processed = 0
            
            # Procesar ensemble para cada chunk y cada cámara
            for chunk_num in range(max_chunk + 1):
                for camera_dir in camera_dirs:
                    camera_id = int(camera_dir.name.replace('camera', ''))
                    
                    if self._process_chunk_ensemble(patient_id, session_id, camera_id, chunk_num):
                        total_processed += 1
            
            logger.info(f"✅ Ensemble completado: {total_processed} chunks procesados para sesión {session_id}")
            
        except Exception as e:
            logger.error(f"Error procesando ensemble de sesión: {e}")
    
    def _find_max_chunk(self, patient_id: str, session_id: str) -> int:
        """Encontrar el chunk máximo en los directorios sin procesar"""
        max_chunk = -1
        session_dir = self.base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}"
        
        camera_dirs = list(session_dir.glob("camera*"))
        for camera_dir in camera_dirs:
            if camera_dir.is_dir():
                chunk_files = list(camera_dir.glob("*.mp4"))
                for chunk_file in chunk_files:
                    try:
                        chunk_num = int(chunk_file.stem)
                        max_chunk = max(max_chunk, chunk_num)
                    except ValueError:
                        continue
        
        return max_chunk
    
    def _process_chunk_ensemble(self, patient_id: str, session_id: str, camera_id: int, chunk_number: int) -> bool:
        """Procesar ensemble para un chunk específico de una cámara"""
        try:
            # Obtener todos los archivos frame_chunk.npy de todos los detectores para este chunk
            all_frame_files = self._get_all_frame_files(patient_id, session_id, camera_id, chunk_number)
            
            if not all_frame_files:
                logger.debug(f"No se encontraron archivos para chunk {chunk_number}, cámara {camera_id}")
                return False
            
            total_processed = 0
            
            # Procesar cada archivo frame_chunk.npy individualmente
            for frame_number in sorted(all_frame_files.keys()):
                frame_data = all_frame_files[frame_number]
                
                # Verificar que tenemos suficientes detectores para este frame
                if len(frame_data) < ensemble_config.min_detectors_required:
                    continue
                
                # Combinar keypoints de todos los detectores para este frame específico
                ensemble_result = self._combine_keypoints(frame_data)
                if ensemble_result is not None:
                    # Guardar resultado individual frame_chunk.npy
                    self._save_single_frame_result(ensemble_result, patient_id, session_id, camera_id, frame_number, chunk_number)
                    total_processed += 1
            
            if total_processed > 0:
                logger.info(f"Ensemble chunk {chunk_number}, cámara {camera_id}: {total_processed} frames procesados")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error procesando ensemble chunk {chunk_number}: {e}")
            return False
    
    def _get_all_frame_files(self, patient_id: str, session_id: str, camera_id: int, chunk_number: int) -> Dict:
        """Obtener todos los archivos frame_chunk.npy organizados por frame number"""
        frame_files = {}
        
        # Los keypoints están en unprocessed
        keypoints_base = self.base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}" / "keypoints2D"
        
        for detector_name in self.detector_instances.keys():
            detector_dir = keypoints_base / detector_name / f"camera{camera_id}" / "coordinates"
            
            if not detector_dir.exists():
                logger.debug(f"No existe directorio para detector {detector_name}: {detector_dir}")
                continue
                
            # Buscar archivos de este chunk - formato: frame_chunk.npy (ej: 120_3.npy)
            chunk_files = list(detector_dir.glob(f"*_{chunk_number}.npy"))
            
            for chunk_file in chunk_files:
                try:
                    # El nombre del archivo es frame_chunk.npy, extraer el frame
                    frame_number = int(chunk_file.stem.split('_')[0])
                    keypoints = np.load(chunk_file)
                    
                    # Organizar por frame number
                    if frame_number not in frame_files:
                        frame_files[frame_number] = {}
                    
                    frame_files[frame_number][detector_name] = keypoints
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error procesando archivo {chunk_file}: {e}")
                    continue
        
        logger.debug(f"Encontrados {len(frame_files)} frames únicos para chunk {chunk_number}, cámara {camera_id}")
        return frame_files

    def _save_single_frame_result(self, keypoints: np.ndarray, patient_id: str, session_id: str, 
                                camera_id: int, frame_number: int, chunk_number: int):
        """Guardar resultado de un frame individual como frame_chunk.npy"""
        try:
            # Guardar en processed, directamente en la carpeta de la cámara (sin subcarpetas)
            output_dir = (self.base_data_dir / "processed" / "2D_keypoints" / 
                         f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Formato: frame_chunk.npy (ej: 120_3.npy)
            output_file = output_dir / f"{frame_number}_{chunk_number}.npy"
            np.save(output_file, keypoints)
            
            logger.debug(f"Frame guardado: {output_file}")
                
        except Exception as e:
            logger.error(f"Error guardando frame {frame_number}_{chunk_number}: {e}")
    
    def _combine_keypoints(self, frame_keypoints: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Combinar keypoints usando las ponderaciones de confianza especificadas
        Solo incluye keypoints con confianza mayor al umbral configurado
        Resultado: array de forma (len(final_keypoint_names), 2)
        """
        try:
            # Array final basado en el número de keypoints finales
            num_final_keypoints = len(self.final_keypoint_names)
            final_keypoints = np.zeros((num_final_keypoints, 2))
            
            # Para cada keypoint final
            for final_kp_idx in range(num_final_keypoints):
                weighted_coords = np.zeros(2)
                total_weight = 0.0
                
                # Buscar este keypoint en todos los detectores
                for detector_name, keypoints in frame_keypoints.items():
                    if detector_name not in self.detector_instances:
                        continue
                    
                    detector_instance = self.detector_instances[detector_name]
                    confidence_weights = detector_instance.ensemble_confidence_weights
                    
                    # Encontrar el índice en este detector que corresponde al keypoint final actual
                    detector_kp_idx = self._find_detector_keypoint_index(final_kp_idx, detector_name)
                    
                    if (detector_kp_idx is not None and 
                        detector_kp_idx < len(confidence_weights) and
                        detector_kp_idx < len(keypoints) and
                        confidence_weights[detector_kp_idx] > 0):
                        
                        # Verificar que el keypoint tenga suficiente confianza real del modelo
                        # keypoints shape: (N, 3) donde [:, 2] es la confianza
                        if keypoints.shape[1] >= 3:
                            model_confidence = keypoints[detector_kp_idx, 2]  # Confianza real del modelo
                            
                            # Solo usar keypoints con confianza mayor al umbral
                            if model_confidence >= ensemble_config.confidence_threshold:
                                ensemble_weight = confidence_weights[detector_kp_idx]
                                coord = keypoints[detector_kp_idx, :2]  # Solo x, y
                                
                                weighted_coords += ensemble_weight * coord
                                total_weight += ensemble_weight
                        else:
                            # Si no hay confianza almacenada, usar el keypoint sin filtro
                            ensemble_weight = confidence_weights[detector_kp_idx]
                            coord = keypoints[detector_kp_idx, :2]  # Solo x, y
                            
                            weighted_coords += ensemble_weight * coord
                            total_weight += ensemble_weight
                
                # Normalizar por peso total
                if total_weight > 0:
                    final_keypoints[final_kp_idx] = weighted_coords / total_weight
                # Si no hay detectores válidos, queda en (0, 0)
            
            return final_keypoints
            
        except Exception as e:
            logger.error(f"Error combinando keypoints: {e}")
            return None
    
    def _find_detector_keypoint_index(self, final_kp_idx: int, detector_name: str) -> Optional[int]:
        """
        Encontrar el índice en un detector específico que corresponde al keypoint final
        """
        try:
            final_keypoint_name = self.final_keypoint_names[final_kp_idx]
            detector_instance = self.detector_instances[detector_name]
            
            # Buscar el nombre en la lista de keypoints del detector
            if final_keypoint_name in detector_instance.keypoints_names:
                return detector_instance.keypoints_names.index(final_keypoint_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error buscando índice de keypoint: {e}")
            return None

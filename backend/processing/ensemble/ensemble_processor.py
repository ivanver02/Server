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
            # Cargar resultados de detectores
            detector_results = self._load_detector_results(patient_id, session_id, camera_id, chunk_number)
            
            if len(detector_results) < ensemble_config.min_detectors_required:
                logger.debug(f"Solo {len(detector_results)} detectores disponibles para chunk {chunk_number}, se requieren al menos {ensemble_config.min_detectors_required}")
                return False
            
            # Obtener todos los frames disponibles
            all_frames = set()
            for detector_frames in detector_results.values():
                all_frames.update(detector_frames.keys())
            
            if not all_frames:
                return False
            
            ensemble_frames = {}
            
            # Procesar ensemble para cada frame
            for global_frame in sorted(all_frames):
                frame_keypoints = {}
                for detector_name, detector_frames in detector_results.items():
                    if global_frame in detector_frames:
                        frame_keypoints[detector_name] = detector_frames[global_frame]
                
                if len(frame_keypoints) >= ensemble_config.min_detectors_required:  # Al menos el mínimo configurado
                    ensemble_result = self._combine_keypoints(frame_keypoints)
                    if ensemble_result is not None:
                        ensemble_frames[global_frame] = ensemble_result
            
            # Guardar resultados
            if ensemble_frames:
                self._save_ensemble_results(ensemble_frames, patient_id, session_id, camera_id, chunk_number)
                logger.debug(f"Ensemble guardado para chunk {chunk_number}, cámara {camera_id}: {len(ensemble_frames)} frames")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error procesando ensemble: {e}")
            return False
    
    def _load_detector_results(self, patient_id: str, session_id: str, camera_id: int, chunk_number: int) -> Dict:
        """Cargar resultados de todos los detectores para un chunk"""
        detector_results = {}
        
        # Los keypoints están en unprocessed, no en processed
        keypoints_base = self.base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}" / "keypoints2D"
        
        for detector_name in self.detector_instances.keys():
            detector_dir = keypoints_base / detector_name / f"camera{camera_id}" / "coordinates"
            
            if not detector_dir.exists():
                logger.debug(f"No existe directorio para detector {detector_name}: {detector_dir}")
                continue
                
            # Buscar archivos de este chunk - formato: frame_chunk.npy (ej: 0_44.npy)
            frame_files = list(detector_dir.glob(f"*_{chunk_number}.npy"))
            
            if frame_files:
                detector_frames = {}
                for frame_file in frame_files:
                    # El nombre del archivo es frame_chunk.npy, extraer el frame
                    global_frame = int(frame_file.stem.split('_')[0])
                    keypoints = np.load(frame_file)
                    detector_frames[global_frame] = keypoints
                    
                detector_results[detector_name] = detector_frames
                logger.debug(f"Cargados {len(detector_frames)} frames para detector {detector_name}, chunk {chunk_number}")
            else:
                logger.debug(f"No se encontraron archivos para detector {detector_name}, chunk {chunk_number} en {detector_dir}")
        
        logger.info(f"Resultados cargados: {len(detector_results)} detectores para chunk {chunk_number}, cámara {camera_id}")
        return detector_results
    
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
    
    def _save_ensemble_results(self, ensemble_frames: Dict, patient_id: str, session_id: str, 
                             camera_id: int, chunk_number: int):
        """Guardar resultados del ensemble como archivos .npy"""
        try:
            # Guardar en processed, siguiendo la estructura estándar
            output_dir = (self.base_data_dir / "processed" / "2D_keypoints" / 
                         f"patient{patient_id}" / f"session{session_id}" / 
                         "ensemble" / f"camera{camera_id}" / "coordinates")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for global_frame, keypoints in ensemble_frames.items():
                output_file = output_dir / f"{global_frame}_{chunk_number}.npy"
                np.save(output_file, keypoints)
                
            logger.info(f"Ensemble guardado: {len(ensemble_frames)} frames en {output_dir}")
                
        except Exception as e:
            logger.error(f"Error guardando resultados de ensemble: {e}")

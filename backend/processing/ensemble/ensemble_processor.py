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
        self.base_data_dir = Path(base_data_dir) # Directorio de la carpeta data/
        
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
        IMPORTANTE (Ahora mismo se cumple, pero cuidado): en todos los modelos que usemos, los nombres asociados al mismo keypoint deben ser los mismos, y deben de venir en el mismo orden. Ejemplo: 'nose' debe ser 'nose' en todos los detectores.
        No debería de suponer mucho problema porque los modelos COCO devuelven los mismos puntos en el mismo orden, igual con los wholebody.
        """
        final_names = []

        final_idx = 0

        # Obtener el número máximo de keypoints entre todos los detectores
        max_keypoints = max(
            len(detector.ensemble_confidence_weights)
            for detector in self.detector_instances.values()
        )

        # Inicializar el diccionario de correspondencia en cada detector
        for detector in self.detector_instances.values():
            detector.final_keypoints_idx = {}

        for kp_idx in range(max_keypoints):
            has_weight = False
            keypoint_name = None

            for detector in self.detector_instances.values():
                if (kp_idx < len(detector.ensemble_confidence_weights) and 
                    detector.ensemble_confidence_weights[kp_idx] > 0):
                    has_weight = True
                    if kp_idx < len(detector.keypoints_names):
                        keypoint_name = detector.keypoints_names[kp_idx]
                        detector.final_keypoints_idx[final_idx] = kp_idx
                    break

            if has_weight and keypoint_name:
                final_names.append(keypoint_name)
                final_idx += 1

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
        
        # Buscar el chunk máximo recibido
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
            logger.debug(f"Chunk {chunk_id} no es el final (max: {max_chunk}), no iniciando ensemble aún")
            return False
            
        session_data['completed_cameras'].add(camera_id)
        logger.info(f"Chunk final completado: patient_id={patient_id}, session_id={session_id}, camera_id={camera_id}, chunk_id={chunk_id}")
        logger.info(f"Cámaras completadas: {len(session_data['completed_cameras'])}/{session_data['cameras_count']}")
        
        # Verificar si todas las cámaras completaron el chunk final
        if len(session_data['completed_cameras']) >= session_data['cameras_count']:
            logger.info(f"¡Todas las cámaras completaron el chunk final! Iniciando ensemble para TODA la sesión {session_id}")
            # Ejecutar ensemble en thread separado
            threading.Thread(
                target=self._process_session_ensemble_async,
                args=(patient_id, session_id, max_chunk),
                daemon=True
            ).start()
            return True
            
        return False

    def _process_session_ensemble_async(self, patient_id: str, session_id: str, max_chunk: int):
        """Procesa el ensemble de forma asíncrona"""
        try:
            self.process_session_ensemble(patient_id, session_id, max_chunk)

            # Limpiar sesión después del ensemble
            if patient_id in self.active_sessions and session_id in self.active_sessions[patient_id]:
                del self.active_sessions[patient_id][session_id]
                if not self.active_sessions[patient_id]:
                    del self.active_sessions[patient_id]
                    
        except Exception as e:
            logger.error(f"Error en ensemble asíncrono: {e}")

    def process_session_ensemble(self, patient_id: str, session_id: str, max_chunk: int):
        """
        Procesar ensemble para toda la sesión cuando todas las cámaras han terminado
        """
        try:
            logger.info(f" Iniciando ensemble para sesión completa: patient{patient_id}/session{session_id}")
            
            # Obtener lista de cámaras y encontrar el chunk máximo
            unprocessed_session = self.base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}"
            camera_dirs = list(unprocessed_session.glob("camera*"))
            
            if not camera_dirs:
                logger.warning("No se encontraron directorios de cámaras")
                return
            
            # Procesar ensemble para cada chunk y cada cámara
            for chunk_num in range(max_chunk + 1):
                logger.info(f" Procesando chunk {chunk_num}/{max_chunk}")
                chunk_processed_count = 0
                
                for camera_dir in camera_dirs:
                    camera_id = int(camera_dir.name.replace('camera', ''))
                    
                    logger.debug(f"Procesando chunk {chunk_num}, cámara {camera_id}")
                    if self._process_chunk_ensemble(patient_id, session_id, camera_id, chunk_num):
                        chunk_processed_count += 1
                        total_processed += 1
                
                logger.info(f" Chunk {chunk_num} completado: {chunk_processed_count}/{len(camera_dirs)} cámaras procesadas")
            
            logger.info(f" Ensemble completado: {total_processed} chunks procesados para sesión {session_id} ({max_chunk + 1} chunks × {len(camera_dirs)} cámaras)")
            
        except Exception as e:
            logger.error(f"Error procesando ensemble de sesión: {e}")
    
    def _process_chunk_ensemble(self, patient_id: str, session_id: str, camera_id: int, chunk_number: int) -> bool:
        """Procesar ensemble para un chunk específico de una cámara"""
        try:
            # Obtener todos los archivos frame_chunk.npy de todos los detectores para este chunk
            # Estructura: {frame_number: {detector_name: {coordinates:coordinate_array, confidence:confidence_array}}}
            all_frame_files = self._get_all_frame_files(patient_id, session_id, camera_id, chunk_number)
            
            if not all_frame_files:
                logger.warning(f"No se encontraron archivos para chunk {chunk_number}, cámara {camera_id}")
                return False
            
            total_processed = 0
            total_skipped = 0
            
            logger.info(f" Procesando {len(all_frame_files)} frames para chunk {chunk_number}, cámara {camera_id}")
            
            # Procesar cada frame para los distintos detectores
            for frame_number in sorted(all_frame_files.keys()):
                # Estructura de frame_data: {detector_name: {coordinates:coordinate_array, confidence:confidence_array}}
                frame_data = all_frame_files[frame_number]
                
                logger.debug(f" Frame {frame_number}: disponible en {len(frame_data)} detectores: {list(frame_data.keys())}")
                
                # Verificar que tenemos suficientes detectores para este frame
                if len(frame_data) < ensemble_config.min_detectors_required:
                    logger.debug(f" Frame {frame_number} omitido: solo {len(frame_data)} detectores (mínimo: {ensemble_config.min_detectors_required})")
                    total_skipped += 1
                    continue
                
                # Combinar keypoints de todos los detectores para este frame específico
                ensemble_result = self._combine_keypoints(frame_data)
                # Estructura de ensemble_result: {coordinates: array(N,2), confidence: array(N)}
                if ensemble_result is not None:
                    # Guardar resultado individual frame_chunk.npy
                    self._save_single_frame_result(ensemble_result, patient_id, session_id, camera_id, frame_number, chunk_number)
                    total_processed += 1
                    logger.debug(f" Frame {frame_number} procesado y guardado")
                else:
                    logger.debug(f" Frame {frame_number} falló en combinación")
                    total_skipped += 1
            
            logger.info(f" Chunk {chunk_number}, cámara {camera_id}: {total_processed} frames procesados, {total_skipped} omitidos")
            return total_processed > 0
            
        except Exception as e:
            logger.error(f"Error procesando ensemble chunk {chunk_number}: {e}")
            return False
    
    def _get_all_frame_files(self, patient_id: str, session_id: str, camera_id: int, chunk_number: int) -> Dict:
        """
        Obtener todos los archivos frame_chunk.npy organizados por frame number
        Devuelve un diccionario {frame_number: {detector_name: {coordinates:coordinate_array, confidence:confidence_array}}}
        """

        frame_files = {}
        keypoints_base = self.base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}" / "keypoints2D"
        logger.debug(f"Buscando keypoints en: {keypoints_base}")

        for detector_name in self.detector_instances.keys():
            coordinates_dir = keypoints_base / detector_name / f"camera{camera_id}" / "coordinates"
            confidence_dir = keypoints_base / detector_name / f"camera{camera_id}" / "confidence"

            if not coordinates_dir.exists():
                logger.debug(f"No existe directorio de coordenadas para detector {detector_name}: {coordinates_dir}")
                continue
            if not confidence_dir.exists():
                logger.debug(f"No existe directorio de confianza para detector {detector_name}: {confidence_dir}")
                continue

            # Buscar todos los archivos de este chunk en coordinates
            chunk_files = list(coordinates_dir.glob(f"*_{chunk_number}.npy"))
            logger.debug(f"Detector {detector_name}, chunk {chunk_number}: encontrados {len(chunk_files)} archivos de coordenadas")

            for chunk_file in chunk_files:
                try:
                    frame_number = int(chunk_file.stem.split('_')[0])
                    coordinates_array = np.load(chunk_file)
                    confidence_file = confidence_dir / chunk_file.name
                    if not confidence_file.exists():
                        logger.debug(f"No existe archivo de confianza para frame {frame_number} en {confidence_file}")
                        continue
                    confidence_array = np.load(confidence_file)

                    if frame_number not in frame_files:
                        frame_files[frame_number] = {}
                    frame_files[frame_number][detector_name] = {
                        'coordinates': coordinates_array,
                        'confidence': confidence_array
                    }
                    logger.debug(f"Cargado frame {frame_number} de {detector_name}: coords shape {coordinates_array.shape}, conf shape {confidence_array.shape}")
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error procesando archivo {chunk_file}: {e}")
                    continue

        logger.info(f"Chunk {chunk_number}, cámara {camera_id}: encontrados {len(frame_files)} frames únicos")
        if len(frame_files) > 0:
            logger.debug(f"Frames encontrados: {sorted(frame_files.keys())}")
        return frame_files

    def _save_single_frame_result(self, ensemble_result: Dict[str, np.ndarray], patient_id: str, session_id: str, 
                                camera_id: int, frame_number: int, chunk_number: int):
        """Guardar resultado de un frame individual en carpetas coordinates y confidence"""
        try:
            output_dir = (self.base_data_dir / "processed" / "2D_keypoints" /
                         f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}")
            coordinates_dir = output_dir / "coordinates"
            confidence_dir = output_dir / "confidence"
            coordinates_dir.mkdir(parents=True, exist_ok=True)
            confidence_dir.mkdir(parents=True, exist_ok=True)

            coordinates_file = coordinates_dir / f"{frame_number}_{chunk_number}.npy"
            confidence_file = confidence_dir / f"{frame_number}_{chunk_number}.npy"

            np.save(coordinates_file, ensemble_result['coordinates'])
            np.save(confidence_file, ensemble_result['confidence'])

            logger.debug(f"Frame guardado: {coordinates_file}, {confidence_file}")
        except Exception as e:
            logger.error(f"Error guardando frame {frame_number}_{chunk_number}: {e}")
    
def _combine_keypoints(self, frame_keypoints: Dict[str, dict]) -> Optional[Dict[str, np.ndarray]]:
        """
        Combina keypoints de todos los detectores usando combinación lineal ponderada por la confianza real y la ponderación de cada detector.
        Devuelve un diccionario: {coordinates: array(N,2), confidence: array(N)}
        """
        try:
            num_final_keypoints = len(self.final_keypoint_names)
            coordinates_array = np.zeros((num_final_keypoints, 2))
            confidence_array = np.zeros(num_final_keypoints)

            for final_kp_idx in range(num_final_keypoints):
                weighted_coords = np.zeros(2)
                weighted_conf = 0.0
                total_weight = 0.0

                for detector_name, detector_data in frame_keypoints.items():
                    if detector_name not in self.detector_instances:
                        continue
                    detector_instance = self.detector_instances[detector_name]
                    confidence_weights = detector_instance.ensemble_confidence_weights
                    detector_kp_idx = detector_instance.final_keypoints_idx.get(final_kp_idx, None)
                    if (
                        detector_kp_idx is not None and
                        detector_kp_idx < len(confidence_weights) and
                        detector_kp_idx < detector_data['coordinates'].shape[0] and
                        detector_kp_idx < detector_data['confidence'].shape[0]
                    ):
                        model_confidence = detector_data['confidence'][detector_kp_idx]
                        ensemble_weight = confidence_weights[detector_kp_idx]
                        if ensemble_weight > 0 and model_confidence > 0:
                            coord = detector_data['coordinates'][detector_kp_idx, :2]
                            weighted_coords += ensemble_weight * model_confidence * coord
                            weighted_conf += ensemble_weight * model_confidence
                            total_weight += ensemble_weight

                if total_weight > 0:
                    coordinates_array[final_kp_idx] = weighted_coords / weighted_conf if weighted_conf > 0 else np.zeros(2)
                    confidence_array[final_kp_idx] = weighted_conf / total_weight
                else:
                    coordinates_array[final_kp_idx] = np.zeros(2)
                    confidence_array[final_kp_idx] = 0.0

            return {'coordinates': coordinates_array, 'confidence': confidence_array}

        except Exception as e:
            logger.error(f"Error combinando keypoints: {e}")
            return None
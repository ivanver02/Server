"""
Coordinador principal del procesamiento de video
Orquesta la sincronización, detección y ensemble learning
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .data import (
    ProcessingSessionResult, MultiCameraResult, SyncConfig
)
from .detectors import MMPoseManager, DetectorFactory
from .processors import MultiCameraProcessor
from .ensemble import EnsembleProcessor
from .synchronization import create_synchronizer_from_videos

logger = logging.getLogger(__name__)


class ProcessingCoordinator:
    """
    Coordinador principal que orquesta todo el pipeline de procesamiento:
    1. Sincronización de videos multi-cámara
    2. Detección de pose con múltiples modelos
    3. Ensemble learning para fusión de resultados
    """
    
    def __init__(self):
        self.detector_manager = MMPoseManager()
        self.multi_camera_processor = MultiCameraProcessor()
        self.ensemble_processor = EnsembleProcessor()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Inicializar todos los componentes del coordinador
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            logger.info("🚀 Inicializando ProcessingCoordinator...")
            
            # 1. Cargar configuración
            from config import processing_config
            
            # 2. Configurar detectores MMPose
            if hasattr(processing_config, 'models_base_path') and processing_config.models_base_path:
                models_path = Path(processing_config.models_base_path)
                if models_path.exists():
                    # Auto-descubrir modelos
                    discovered = self.detector_manager.auto_discover_models(models_path)
                    logger.info(f"Modelos auto-descubiertos: {discovered}")
                else:
                    logger.warning(f"Directorio de modelos no existe: {models_path}")
            
            # Registrar modelos específicos si están configurados
            all_models = []
            if hasattr(processing_config, 'coco_models'):
                all_models.extend(processing_config.coco_models)
            if hasattr(processing_config, 'extended_models'):
                all_models.extend(processing_config.extended_models)
            
            for model_name in all_models:
                if hasattr(processing_config, 'models_base_path'):
                    model_path = Path(processing_config.models_base_path) / model_name
                    if model_path.exists():
                        if self.detector_manager.register_mmpose_model(model_name, model_path):
                            logger.info(f"Modelo registrado: {model_name}")
                        else:
                            logger.warning(f"Error registrando modelo: {model_name}")
                    else:
                        logger.warning(f"Modelo no encontrado: {model_path}")
            
            # 3. Inicializar detectores
            if not self.detector_manager.initialize_all():
                logger.error("Error inicializando detectores")
                return False
            
            # 4. Inicializar procesador multi-cámara
            if not self.multi_camera_processor.initialize(self.detector_manager):
                logger.error("Error inicializando procesador multi-cámara")
                return False
            
            self.is_initialized = True
            active_detectors = self.detector_manager.get_active_detectors()
            logger.info(f"✅ ProcessingCoordinator inicializado con {len(active_detectors)} detectores activos")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando ProcessingCoordinator: {e}")
            return False
    
    def process_chunk_videos(self, 
                           patient_id: str,
                           session_id: str,
                           chunk_number: int,
                           video_paths: Dict[int, Path],
                           sync_config: Optional[SyncConfig] = None) -> MultiCameraResult:
        """
        Procesar chunk con múltiples videos sincronizados
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            chunk_number: Número del chunk
            video_paths: Diccionario {camera_id: video_path}
            sync_config: Configuración de sincronización
            
        Returns:
            Resultado del procesamiento del chunk
        """
        if not self.is_initialized:
            return MultiCameraResult(
                success=False,
                patient_id=patient_id,
                session_id=session_id,
                chunk_number=chunk_number,
                sync_frame_results=[],
                processing_time=0,
                total_frames=0,
                camera_videos={k: str(v) for k, v in video_paths.items()},
                errors=["Coordinador no inicializado"]
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"🎬 Procesando chunk {chunk_number} - "
                       f"Paciente: {patient_id}, Sesión: {session_id}")
            logger.info(f"Videos: {list(video_paths.keys())}")
            
            # Usar configuración por defecto si no se proporciona
            if sync_config is None:
                from config import processing_config
                sync_config = SyncConfig(
                    target_fps=processing_config.target_fps,
                    frame_interval=getattr(processing_config, 'frame_interval', 1),
                    sync_tolerance=getattr(processing_config, 'sync_tolerance', 0.1),
                    quality_threshold=getattr(processing_config, 'quality_threshold', 0.8)
                )
            
            # Procesar con multi-cámara
            result = self.multi_camera_processor.process_synchronized_videos(
                video_paths=video_paths,
                patient_id=patient_id,
                session_id=session_id,
                chunk_number=chunk_number,
                sync_config=sync_config
            )
            
            if not result.success:
                return result
            
            # Guardar resultados de keypoints 2D
            self._save_keypoints_2d(result)
            
            # Aplicar ensemble learning si hay múltiples detectores
            active_detectors = self.detector_manager.get_active_detectors()
            if len(active_detectors) > 1:
                logger.info(f"🎯 Aplicando ensemble learning con {len(active_detectors)} detectores")
                
                ensemble_result = self._apply_ensemble_to_chunk(result)
                if ensemble_result:
                    # Actualizar resultado con información de ensemble
                    result.sync_info['ensemble_applied'] = True
                    result.sync_info['ensemble_result'] = ensemble_result
                else:
                    logger.warning("Error aplicando ensemble learning")
            
            total_time = time.time() - start_time
            result.processing_time = total_time
            
            logger.info(f"✅ Chunk {chunk_number} procesado en {total_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Error procesando chunk {chunk_number}: {e}"
            logger.error(error_msg)
            
            return MultiCameraResult(
                success=False,
                patient_id=patient_id,
                session_id=session_id,
                chunk_number=chunk_number,
                sync_frame_results=[],
                processing_time=time.time() - start_time,
                total_frames=0,
                camera_videos={k: str(v) for k, v in video_paths.items()},
                errors=[error_msg]
            )
    
    def _apply_ensemble_to_chunk(self, chunk_result: MultiCameraResult) -> Optional[Dict[str, Any]]:
        """
        Aplicar ensemble learning a los resultados de un chunk
        """
        try:
            # Extraer keypoints de todos los frames y detectores
            ensemble_results = {}
            
            for sync_frame_result in chunk_result.sync_frame_results:
                frame_number = sync_frame_result.frame_number
                
                for camera_id, frame_result in sync_frame_result.camera_results.items():
                    # Recopilar keypoints de todos los detectores para este frame/cámara
                    detector_keypoints = {}
                    detector_scores = {}
                    
                    for detector_name, keypoint_result in frame_result.keypoint_results.items():
                        if keypoint_result.success and keypoint_result.keypoints is not None:
                            detector_keypoints[detector_name] = keypoint_result.keypoints
                            detector_scores[detector_name] = keypoint_result.scores
                    
                    # Aplicar ensemble si hay múltiples detectores
                    if len(detector_keypoints) > 1:
                        ensemble_result = self.ensemble_processor.fuse_keypoints(
                            detector_keypoints, detector_scores
                        )
                        
                        if ensemble_result:
                            ensemble_results[f"frame_{frame_number}_camera_{camera_id}"] = {
                                'keypoints_2d': ensemble_result.keypoints_2d.tolist(),
                                'confidence_scores': ensemble_result.confidence_scores.tolist(),
                                'model_contributions': ensemble_result.model_contributions
                            }
            
            return {
                'total_ensembles': len(ensemble_results),
                'ensemble_results': ensemble_results,
                'processing_info': {
                    'ensemble_method': 'weighted_confidence',
                    'confidence_threshold': self.ensemble_processor.confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error aplicando ensemble: {e}")
            return None
    
    def _count_total_keypoints(self, result: MultiCameraResult) -> int:
        """Contar total de keypoints detectados"""
        total = 0
        for sync_frame_result in result.sync_frame_results:
            for frame_result in sync_frame_result.camera_results.values():
                for keypoint_result in frame_result.keypoint_results.values():
                    if keypoint_result.success and keypoint_result.keypoints is not None:
                        total += len(keypoint_result.keypoints)
        return total
    
    def _save_keypoints_2d(self, result: MultiCameraResult) -> List[str]:
        """
        Guardar keypoints 2D en formato .npy para triangulación rápida
        
        Args:
            result: Resultado del procesamiento multi-cámara
            
        Returns:
            Lista de rutas de archivos guardados
        """
        try:
            from config import data_config
            
            saved_files = []
            base_path = data_config.base_data_dir / f"patient{result.patient_id}" / f"session{result.session_id}"
            
            for sync_frame_result in result.sync_frame_results:
                frame_number = sync_frame_result.frame_number
                timestamp = sync_frame_result.timestamp
                
                for camera_id, frame_result in sync_frame_result.camera_results.items():
                    # Crear directorio para esta cámara (estructura compatible con triangulador)
                    camera_dir = base_path / f"camera{camera_id}"
                    camera_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Guardar keypoints de cada detector por separado
                    for detector_name, keypoint_result in frame_result.keypoint_results.items():
                        if keypoint_result.success and keypoint_result.keypoints is not None:
                            
                            # Nombre de archivo compatible con triangulador
                            # global_frame = chunk_number * frames_per_chunk + frame_number
                            global_frame = result.chunk_number * 1000 + frame_number  # Asumiendo max 1000 frames por chunk
                            
                            # Guardar keypoints en .npy (para triangulación rápida)
                            keypoints_file = camera_dir / f"{global_frame}_{detector_name}.npy"
                            np.save(keypoints_file, keypoint_result.keypoints)
                            saved_files.append(str(keypoints_file))
                            
                            # Guardar confianzas en .npy
                            if keypoint_result.scores is not None:
                                confidence_file = camera_dir / f"{global_frame}_{detector_name}_confidence.npy"
                                np.save(confidence_file, keypoint_result.scores)
                                saved_files.append(str(confidence_file))
                            
                            # Guardar metadata en .json (para debugging y análisis)
                            metadata = {
                                'patient_id': result.patient_id,
                                'session_id': result.session_id,
                                'chunk_number': result.chunk_number,
                                'camera_id': camera_id,
                                'frame_number': frame_number,
                                'global_frame': global_frame,
                                'timestamp': timestamp,
                                'detector_name': detector_name,
                                'keypoints_shape': keypoint_result.keypoints.shape,
                                'num_keypoints': len(keypoint_result.keypoints),
                                'bbox': keypoint_result.bbox.tolist() if keypoint_result.bbox is not None else None,
                                'processing_time': keypoint_result.processing_time,
                                'sync_quality': sync_frame_result.sync_quality,
                                'metadata': keypoint_result.metadata or {}
                            }
                            
                            metadata_file = camera_dir / f"{global_frame}_{detector_name}_metadata.json"
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            saved_files.append(str(metadata_file))
            
            logger.info(f"💾 Guardados {len(saved_files)} archivos de keypoints 2D para chunk {result.chunk_number}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error guardando keypoints 2D: {e}")
            return []
    
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado completo del coordinador"""
        return {
            'is_initialized': self.is_initialized,
            'detector_manager': self.detector_manager.get_status() if self.detector_manager else None,
            'multi_camera_processor': self.multi_camera_processor.get_status() if self.multi_camera_processor else None
        }
    
    def cleanup(self):
        """Limpiar todos los recursos"""
        try:
            logger.info("🧹 Limpiando ProcessingCoordinator...")
            
            # Limpiar componentes
            if self.detector_manager:
                self.detector_manager.cleanup_all()
            
            if self.multi_camera_processor:
                self.multi_camera_processor.cleanup()
            
            # Cerrar executor
            self.executor.shutdown(wait=True)
            
            self.is_initialized = False
            logger.info("✅ ProcessingCoordinator limpiado")
            
        except Exception as e:
            logger.error(f"❌ Error limpiando ProcessingCoordinator: {e}")


# Instancia global del coordinador
processing_coordinator = ProcessingCoordinator()

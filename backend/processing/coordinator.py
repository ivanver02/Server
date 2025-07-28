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
    ProcessingSessionResult, MultiCameraResult
)
from .detectors import (
    VitPoseDetector, HRNetW48Detector, WholeBodyDetector, RTMPoseDetector, ResNet50RLEDetector
)
from .processors import MultiCameraProcessor
from .ensemble import EnsembleProcessor
from .synchronization import create_synchronizer_from_videos

logger = logging.getLogger(__name__)


class ProcessingCoordinator:
    """
    Coordinador principal que orquesta todo el pipeline de procesamiento:
    1. Sincronización de videos multi-cámara
    2. Detección de pose con múltiples detectores MMPose
    3. Ensemble learning para fusión de resultados
    """
    
    def __init__(self):
        # Inicializar los cinco detectores específicos
        self.detectors = [
            VitPoseDetector(),
            HRNetW48Detector(),
            WholeBodyDetector(),
            RTMPoseDetector(),
            ResNet50RLEDetector()
        ]
        self.multi_camera_processor = MultiCameraProcessor()
        self.ensemble_processor = EnsembleProcessor()
        
        # Usar configuración centralizada para max_workers
        from config import processing_config
        self.executor = ThreadPoolExecutor(max_workers=processing_config.max_workers)
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Inicializar todos los componentes del coordinador
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            logger.info("Inicializando ProcessingCoordinator...")
            
            # Inicializar detectores
            initialized_detectors = 0
            for detector in self.detectors:
                if detector.initialize():
                    initialized_detectors += 1
                    logger.info(f"Detector {detector.name} inicializado correctamente")
                else:
                    logger.error(f"Error inicializando detector {detector.name}")
            
            if initialized_detectors == 0:
                logger.error("No se pudo inicializar ningún detector")
                return False
            
            # Inicializar procesador multi-cámara
            if not self.multi_camera_processor.initialize(self.detectors):
                logger.error("Error inicializando procesador multi-cámara")
                return False
            
            self.is_initialized = True
            logger.info(f"ProcessingCoordinator inicializado con {initialized_detectors}/{len(self.detectors)} detectores activos")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando ProcessingCoordinator: {e}")
            return False
    
    def process_chunk_videos(self, 
                           patient_id: str,
                           session_id: str,
                           chunk_number: int,
                           video_paths: Dict[int, Path]) -> MultiCameraResult:
        """
        Procesar chunk con múltiples videos sincronizados
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            chunk_number: Número del chunk
            video_paths: Diccionario {camera_id: video_path}
            
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
            
            # Procesar con multi-cámara
            result = self.multi_camera_processor.process_synchronized_videos(
                video_paths=video_paths,
                patient_id=patient_id,
                session_id=session_id,
                chunk_number=chunk_number
            )
            
            if not result.success:
                return result
            
            # Keypoints 2D ya se guardaron directamente en MultiCameraProcessor
            
            total_time = time.time() - start_time
            result.processing_time = total_time
            
            logger.info(f"Chunk {chunk_number} procesado en {total_time:.2f}s")
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

    
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado completo del coordinador"""
        return {
            'is_initialized': self.is_initialized,
            'detectors': [detector.get_status() for detector in self.detectors],
            'multi_camera_processor': self.multi_camera_processor.get_status() if self.multi_camera_processor else None
        }
    
    def cleanup(self):
        """Limpiar todos los recursos"""
        try:
            logger.info("🧹 Limpiando ProcessingCoordinator...")
            
            # Limpiar detectores
            for detector in self.detectors:
                detector.cleanup()
            
            if self.multi_camera_processor:
                self.multi_camera_processor.cleanup()
            
            # Cerrar executor
            self.executor.shutdown(wait=True)
            
            self.is_initialized = False
            logger.info("ProcessingCoordinator limpiado")
            
        except Exception as e:
            logger.error(f"Error limpiando ProcessingCoordinator: {e}")


# Instancia global del coordinador
processing_coordinator = ProcessingCoordinator()

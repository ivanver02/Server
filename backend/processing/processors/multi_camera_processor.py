"""
Procesador multi-cámara simplificado que integra sincronización y detección de pose
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data import (
    MultiCameraResult, SyncFrameResult, FrameResult,
    VideoInfo, SyncConfig
)
from ..detectors import BasePoseDetector
from ..synchronization import VideoSynchronizer, create_synchronizer_from_videos
from ..utils import save_keypoints_2d_frame, save_frame_metadata

logger = logging.getLogger(__name__)


class MultiCameraProcessor:
    """
    Procesador que coordina la sincronización de videos y detección de pose
    en múltiples cámaras simultáneamente, guardando directamente en archivos .npy
    """
    
    def __init__(self):
        # Usar configuración centralizada para max_workers
        from config import processing_config
        self.detectors: List[BasePoseDetector] = []
        self.max_workers = processing_config.max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.is_initialized = False
    
    def initialize(self, detectors: List[BasePoseDetector]) -> bool:
        """
        Inicializar procesador con lista de detectores
        
        Args:
            detectors: Lista de detectores inicializados
            
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Filtrar solo detectores inicializados
            active_detectors = [d for d in detectors if d.is_initialized]
            
            if not active_detectors:
                logger.error("Debe haber al menos un detector inicializado")
                return False
            
            self.detectors = active_detectors
            self.is_initialized = True
            
            logger.info(f"MultiCameraProcessor inicializado con {len(active_detectors)} detectores")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando MultiCameraProcessor: {e}")
            return False
    
    def process_synchronized_videos(self,
                                  video_paths: Dict[int, Path],
                                  patient_id: str,
                                  session_id: str,
                                  chunk_number: int) -> MultiCameraResult:
        """
        Procesar múltiples videos sincronizados
        
        Args:
            video_paths: Diccionario {camera_id: video_path}
            patient_id: ID del paciente
            session_id: ID de la sesión
            chunk_number: Número del chunk
            
        Returns:
            Resultado del procesamiento multi-cámara
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
                errors=["Procesador no inicializado"]
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"🎬 Iniciando procesamiento sincronizado de {len(video_paths)} videos")
            
            # Crear sincronizador
            synchronizer = create_synchronizer_from_videos(
                video_paths=video_paths
            )
            
            if not synchronizer.initialize_sync():
                return MultiCameraResult(
                    success=False,
                    patient_id=patient_id,
                    session_id=session_id,
                    chunk_number=chunk_number,
                    sync_frame_results=[],
                    processing_time=time.time() - start_time,
                    total_frames=0,
                    camera_videos={k: str(v) for k, v in video_paths.items()},
                    errors=["Error inicializando sincronización"]
                )
            
            # Procesar frames sincronizados
            sync_frame_results = self._process_synchronized_frames(
                synchronizer, patient_id, session_id, chunk_number
            )
            
            total_time = time.time() - start_time
            
            return MultiCameraResult(
                success=True,
                patient_id=patient_id,
                session_id=session_id,
                chunk_number=chunk_number,
                sync_frame_results=sync_frame_results,
                processing_time=total_time,
                total_frames=len(sync_frame_results),
                camera_videos={k: str(v) for k, v in video_paths.items()},
                sync_info={
                    'synchronizer_info': synchronizer.get_sync_info(),
                    'detectors_used': [d.name for d in self.detectors]
                }
            )
            
        except Exception as e:
            error_msg = f"Error en procesamiento multi-cámara: {e}"
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
    
    def _process_synchronized_frames(self,
                                   synchronizer: VideoSynchronizer,
                                   patient_id: str,
                                   session_id: str,
                                   chunk_number: int) -> List[SyncFrameResult]:
        """
        Procesar frames sincronizados con detección de pose y guardado directo
        """
        sync_frame_results = []
        
        logger.info(f"🔍 Procesando frames con {len(self.detectors)} detectores")
        
        try:
            frame_count = 0
            for sync_frame in synchronizer.iterate_synchronized_frames():
                frame_count += 1
                
                # Procesar cada cámara del frame sincronizado
                camera_results = {}
                
                for camera_id in sync_frame.available_cameras:
                    frame = sync_frame.camera_frames[camera_id]
                    
                    # Detectar keypoints con todos los detectores activos
                    detector_results = {}
                    
                    for detector in self.detectors:
                        if detector.is_initialized:
                            keypoints, scores = detector.detect_frame(frame)
                            if keypoints is not None and scores is not None:
                                detector_results[detector.name] = {
                                    'keypoints': keypoints,
                                    'scores': scores
                                }
                    
                    # Guardar directamente usando funciones auxiliares
                    if detector_results:
                        global_frame = chunk_number * 1000 + frame_count
                        
                        for detector_name, data in detector_results.items():
                            try:
                                save_keypoints_2d_frame(
                                    keypoints=data['keypoints'],
                                    confidence=data['scores'],
                                    detector_name=detector_name,
                                    camera_id=camera_id,
                                    global_frame=global_frame,
                                    patient_id=patient_id,
                                    session_id=session_id
                                )
                            except Exception as e:
                                logger.error(f"Error guardando keypoints detector {detector_name}: {e}")
                    
                    # Crear resultado del frame con estructura correcta
                    frame_result = FrameResult(
                        frame_number=frame_count,
                        timestamp=sync_frame.timestamp,
                        camera_id=camera_id,
                        frame_processed=len(detector_results) > 0,
                        detectors_used=list(detector_results.keys()),
                        num_detections=len(detector_results)
                    )
                    camera_results[camera_id] = frame_result
                
                # Crear resultado del frame sincronizado simplificado
                sync_frame_result = SyncFrameResult(
                    frame_number=frame_count,
                    timestamp=sync_frame.timestamp,
                    camera_results=camera_results,
                    sync_quality=sync_frame.sync_quality
                )
                
                sync_frame_results.append(sync_frame_result)
                
                if frame_count % 100 == 0:
                    logger.info(f"Procesados {frame_count} frames...")
            
            logger.info(f"✅ Procesamiento completado: {len(sync_frame_results)} frames sincronizados")
            
        except Exception as e:
            logger.error(f"Error procesando frames: {e}")
            raise
        
        return sync_frame_results
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del procesador"""
        return {
            'is_initialized': self.is_initialized,
            'detectors': [detector.get_status() for detector in self.detectors],
            'max_workers': self.max_workers
        }
    
    def cleanup(self):
        """Limpiar recursos del procesador"""
        try:
            self.executor.shutdown(wait=True)
            for detector in self.detectors:
                detector.cleanup()
            self.is_initialized = False
            logger.info("MultiCameraProcessor limpiado")
        except Exception as e:
            logger.error(f"Error limpiando MultiCameraProcessor: {e}")

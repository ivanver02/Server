"""
Procesador multi-cámara que integra sincronización y detección de pose
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data import (
    MultiCameraResult, FrameResult, SyncFrameResult, 
    SyncConfig, VideoInfo
)
from ..detectors import BaseDetectorManager
from ..synchronization import VideoSynchronizer, create_synchronizer_from_videos

logger = logging.getLogger(__name__)


class MultiCameraProcessor:
    """
    Procesador que coordina la sincronización de videos y detección de pose
    en múltiples cámaras simultáneamente
    """
    
    def __init__(self, max_workers: int = 4):
        self.detector_manager: Optional[BaseDetectorManager] = None
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_initialized = False
    
    def initialize(self, detector_manager: BaseDetectorManager) -> bool:
        """
        Inicializar procesador con manager de detectores
        
        Args:
            detector_manager: Manager de detectores inicializado
            
        Returns:
            True si se inicializó correctamente
        """
        try:
            if not detector_manager.get_active_detectors():
                logger.error("El detector manager debe tener al menos un detector activo")
                return False
            
            self.detector_manager = detector_manager
            self.is_initialized = True
            
            logger.info(f"MultiCameraProcessor inicializado con {len(detector_manager.get_active_detectors())} detectores")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando MultiCameraProcessor: {e}")
            return False
    
    def process_synchronized_videos(self, 
                                   video_paths: Dict[int, Path],
                                   patient_id: str,
                                   session_id: str, 
                                   chunk_number: int,
                                   sync_config: Optional[SyncConfig] = None) -> MultiCameraResult:
        """
        Procesar múltiples videos sincronizados
        
        Args:
            video_paths: Diccionario {camera_id: video_path}
            patient_id: ID del paciente
            session_id: ID de la sesión
            chunk_number: Número del chunk
            sync_config: Configuración de sincronización
            
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
        errors = []
        
        try:
            logger.info(f"🎬 Procesando videos sincronizados - "
                       f"Paciente: {patient_id}, Sesión: {session_id}, Chunk: {chunk_number}")
            logger.info(f"Cámaras: {list(video_paths.keys())}")
            
            # 1. Crear y configurar sincronizador
            synchronizer = create_synchronizer_from_videos(video_paths, sync_config)
            
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
            
            # 2. Procesar frames sincronizados
            sync_frame_results = self._process_synchronized_frames(
                synchronizer, patient_id, session_id, chunk_number
            )
            
            # 3. Limpiar sincronizador
            sync_info = synchronizer.get_sync_info()
            synchronizer.cleanup()
            
            processing_time = time.time() - start_time
            success = len(sync_frame_results) > 0 and all(
                sfr.camera_results for sfr in sync_frame_results
            )
            
            if not success:
                errors.append("No se procesaron frames correctamente")
            
            result = MultiCameraResult(
                success=success,
                patient_id=patient_id,
                session_id=session_id,
                chunk_number=chunk_number,
                sync_frame_results=sync_frame_results,
                processing_time=processing_time,
                total_frames=len(sync_frame_results),
                camera_videos={k: str(v) for k, v in video_paths.items()},
                sync_info=sync_info,
                errors=errors
            )
            
            if success:
                logger.info(f"✅ Videos procesados exitosamente: {len(sync_frame_results)} frames en {processing_time:.2f}s")
            else:
                logger.error(f"❌ Error procesando videos: {errors}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error crítico procesando videos: {e}"
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
        Procesar frames sincronizados con detección de pose
        """
        sync_frame_results = []
        active_detectors = self.detector_manager.get_active_detectors()
        
        logger.info(f"🔍 Procesando frames con detectores: {active_detectors}")
        
        try:
            frame_count = 0
            for sync_frame in synchronizer.iterate_synchronized_frames():
                frame_count += 1
                
                # Procesar cada cámara del frame sincronizado
                camera_results = {}
                
                for camera_id in sync_frame.available_cameras:
                    frame = sync_frame.camera_frames[camera_id]
                    
                    # Detectar keypoints con todos los detectores activos
                    keypoint_results = {}
                    
                    for detector_name in active_detectors:
                        detector = self.detector_manager.get_detector(detector_name)
                        if detector:
                            result = detector.detect_frame(frame)
                            keypoint_results[detector_name] = result
                    
                    # Crear resultado del frame
                    frame_result = FrameResult(
                        frame_number=sync_frame.frame_number,
                        timestamp=sync_frame.timestamp,
                        camera_id=camera_id,
                        keypoint_results=keypoint_results,
                        processing_time=sum(kr.processing_time for kr in keypoint_results.values()),
                        success=any(kr.success for kr in keypoint_results.values())
                    )
                    
                    camera_results[camera_id] = frame_result
                
                # Crear resultado del frame sincronizado
                sync_frame_result = SyncFrameResult(
                    frame_number=sync_frame.frame_number,
                    timestamp=sync_frame.timestamp,
                    camera_results=camera_results,
                    available_cameras=sync_frame.available_cameras,
                    sync_quality=sync_frame.sync_quality,
                    processing_time=sum(cr.processing_time for cr in camera_results.values())
                )
                
                sync_frame_results.append(sync_frame_result)
                
                # Log progreso cada 50 frames
                if frame_count % 50 == 0:
                    logger.info(f"Procesados {frame_count} frames sincronizados...")
            
            logger.info(f"Completado procesamiento de {len(sync_frame_results)} frames sincronizados")
            
        except Exception as e:
            logger.error(f"Error procesando frames sincronizados: {e}")
        
        return sync_frame_results
    
    def _process_frame_parallel(self, 
                              frame: Any, 
                              camera_id: int,
                              detector_names: List[str]) -> FrameResult:
        """
        Procesar frame individual con múltiples detectores en paralelo
        """
        # Esta función se puede usar para procesamiento paralelo futuro
        # Por ahora mantenemos procesamiento secuencial para simplicidad
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del procesador"""
        return {
            'is_initialized': self.is_initialized,
            'max_workers': self.max_workers,
            'detector_manager_status': (
                self.detector_manager.get_available_detectors() 
                if self.detector_manager else None
            )
        }
    
    def cleanup(self):
        """Limpiar recursos del procesador"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            self.is_initialized = False
            logger.info("MultiCameraProcessor limpiado")
            
        except Exception as e:
            logger.error(f"Error limpiando MultiCameraProcessor: {e}")

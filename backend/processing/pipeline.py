"""
Pipeline principal del sistema de análisis de marcha
Documenta y centraliza el flujo completo de procesamiento

FLUJO DE DATOS DE KEYPOINTS 2D:
1. MMPose detectores devuelven: (keypoints: np.ndarray(N,2), scores: np.ndarray(N,))
2. Se guardan por separado en estructura organizada:
   data/patient{}/session{}/camera{}/
   ├── keypoints/{global_frame}_{detector}.npy  # Coordenadas (N,2)
   └── confidence/{global_frame}_{detector}.npy # Confianza (N,)

DETECTORES ACTIVOS:
- VitPose: Detector principal de alta precisión
- HRNet-W48: Detector robusto para condiciones difíciles  
- WholeBody: Detector de cuerpo completo con manos y cara
- RTMPose: Detector rápido en tiempo real
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .coordinator import ProcessingCoordinator
from .data import MultiCameraResult
from .utils import get_session_summary, get_available_frames, get_available_detectors

logger = logging.getLogger(__name__)


class VideoProcessingPipeline:
    """
    Pipeline principal que coordina todo el flujo de procesamiento de video:
    
    FLUJO COMPLETO DE UN CHUNK:
    1. Recepción de chunks por cámara (/api/chunks/receive)
    2. Verificación de chunks completos (todas las cámaras)
    3. Sincronización temporal de videos multi-cámara
    4. Detección de pose frame a frame con 4 detectores MMPose
    5. Guardado directo por separado:
       - Coordenadas: keypoints/{frame}_{detector}.npy
       - Confianza: confidence/{frame}_{detector}.npy
    6. Activación de triangulación 3D cuando hay suficientes datos
    7. Reconstrucción final y análisis biomecánico
    
    FORMATO DE DATOS:
    - Input MMPose: (keypoints: np.ndarray(N,2), scores: np.ndarray(N,))
    - Output archivo: keypoints y confidence guardados por separado
    - No se mantienen KeypointResult en memoria para 2D (solo se guardan)
    """
    
    def __init__(self):
        self.coordinator = ProcessingCoordinator()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Inicializar el pipeline completo
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            logger.info("Inicializando pipeline de procesamiento de video...")
            
            # Inicializar coordinador
            if not self.coordinator.initialize():
                logger.error("Error inicializando coordinador")
                return False
            
            self.is_initialized = True
            logger.info("Pipeline inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando pipeline: {e}")
            return False
    
    def process_chunk_synchronized(self,
                                 patient_id: str,
                                 session_id: str,
                                 chunk_number: int,
                                 video_paths: Dict[int, Path]) -> MultiCameraResult:
        """
        Procesar un chunk completo con todas las cámaras
        
        FLUJO DETALLADO DE PROCESAMIENTO:
        1. Validar que tenemos videos de todas las cámaras esperadas
        2. Crear sincronizador usando timestamps y frame rates
        3. Para cada frame sincronizado:
           a. Extraer frame de cada cámara en el timestamp
           b. Detectar keypoints con los 4 detectores MMPose:
              - VitPose: keypoints(N,2), scores(N,)
              - HRNet-W48: keypoints(N,2), scores(N,)
              - WholeBody: keypoints(N,2), scores(N,)
              - RTMPose: keypoints(N,2), scores(N,)
           c. Guardar cada detector por separado:
              - keypoints/{global_frame}_{detector}.npy
              - confidence/{global_frame}_{detector}.npy
           d. Actualizar resultado del frame (solo metadatos)
        4. Retornar resultado del procesamiento con estadísticas
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión  
            chunk_number: Número del chunk
            video_paths: Dict {camera_id: video_path}
            
        Returns:
            Resultado del procesamiento multi-cámara
        """
        if not self.is_initialized:
            logger.error("Pipeline no inicializado")
            return MultiCameraResult(
                success=False,
                patient_id=patient_id,
                session_id=session_id,
                chunk_number=chunk_number,
                sync_frame_results=[],
                processing_time=0,
                total_frames=0,
                camera_videos={k: str(v) for k, v in video_paths.items()},
                errors=["Pipeline no inicializado"]
            )
        
        logger.info(f"📹 Procesando chunk {chunk_number} con {len(video_paths)} cámaras")
        
        # Delegar al coordinador
        return self.coordinator.process_chunk_videos(
            patient_id=patient_id,
            session_id=session_id,
            chunk_number=chunk_number,
            video_paths=video_paths
        )
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del pipeline
        
        Returns:
            Estado detallado de todos los componentes
        """
        return {
            'pipeline_initialized': self.is_initialized,
            'coordinator_status': self.coordinator.get_status() if self.is_initialized else None,
            'data_structure': {
                'keypoints_format': 'np.ndarray(N, 2) - coordenadas 2D',
                'confidence_format': 'np.ndarray(N,) - confianza por keypoint',
                'storage_structure': 'keypoints/ y confidence/ separados',
                'filename_pattern': '{global_frame}_{detector_name}.npy'
            },
            'detectors': {
                'active': ['VitPose', 'HRNet-W48', 'WholeBody', 'RTMPose'],
                'output_format': 'tuple(keypoints, scores) directo de MMPose',
                'keypoint_count': 'Variable por detector (17+ keypoints)'
            },
            'processing_flow': {
                'sync_method': 'Timestamp-based with FPS alignment',
                'detection_method': 'Parallel execution on all detectors',
                'storage_method': 'Direct .npy saving (no in-memory buffers)',
                'triangulation_trigger': 'Automatic when sufficient 2D data available'
            }
        }
    
    def get_session_analysis(self, patient_id: str, session_id: str) -> Dict[str, Any]:
        """
        Obtener análisis completo de una sesión procesada
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            
        Returns:
            Análisis detallado de la sesión
        """
        try:
            # Obtener resumen básico
            summary = get_session_summary(patient_id, session_id)
            
            if 'error' in summary:
                return summary
            
            # Análisis por cámara
            camera_analysis = {}
            for camera_id, camera_data in summary['cameras'].items():
                frames = get_available_frames(camera_id, patient_id, session_id)
                detectors = get_available_detectors(camera_id, patient_id, session_id)
                
                camera_analysis[camera_id] = {
                    'total_frames': len(frames),
                    'frame_range': camera_data['frame_range'],
                    'detectors_available': detectors,
                    'detection_coverage': {
                        detector: len(get_available_frames(camera_id, patient_id, session_id, detector))
                        for detector in detectors
                    }
                }
            
            return {
                'session_summary': summary,
                'camera_analysis': camera_analysis,
                'processing_status': 'complete' if summary['total_frames'] > 0 else 'pending',
                'ready_for_3d': len(summary['cameras']) >= 2 and summary['total_frames'] > 0
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de sesión: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Limpiar todos los recursos del pipeline"""
        try:
            if self.is_initialized:
                self.coordinator.cleanup()
                self.is_initialized = False
                logger.info("Pipeline limpiado correctamente")
        except Exception as e:
            logger.error(f"Error limpiando pipeline: {e}")


# Instancia global del pipeline
video_pipeline = VideoProcessingPipeline()


def initialize_pipeline() -> bool:
    """
    Función helper para inicializar el pipeline global
    
    Returns:
        True si se inicializó correctamente
    """
    return video_pipeline.initialize()


def process_chunk(patient_id: str,
                 session_id: str, 
                 chunk_number: int,
                 video_paths: Dict[int, Path]) -> MultiCameraResult:
    """
    Función helper para procesar un chunk usando el pipeline global
    
    Args:
        patient_id: ID del paciente
        session_id: ID de la sesión
        chunk_number: Número del chunk  
        video_paths: Dict {camera_id: video_path}
        
    Returns:
        Resultado del procesamiento
    """
    return video_pipeline.process_chunk_synchronized(
        patient_id=patient_id,
        session_id=session_id,
        chunk_number=chunk_number,
        video_paths=video_paths
    )


def get_pipeline_status() -> Dict[str, Any]:
    """
    Función helper para obtener estado del pipeline global
    
    Returns:
        Estado del pipeline
    """
    return video_pipeline.get_processing_status()

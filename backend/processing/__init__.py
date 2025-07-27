"""
Sistema de procesamiento de video para análisis de gonartrosis
Arquitectura simplificada con detectores específicos
"""

# Importar coordinador principal
from .coordinator import ProcessingCoordinator, processing_coordinator

# Importar pipeline principal
from .pipeline import VideoProcessingPipeline, video_pipeline, initialize_pipeline, process_chunk

# Importar estructuras de datos
from .data import (
    FrameResult, SyncFrameResult,
    VideoProcessingResult, MultiCameraResult, ProcessingSessionResult,
    VideoInfo, SyncFrame, SyncConfig
)

# Importar detectores
from .detectors import (
    BasePoseDetector,
    VitPoseDetector, HRNetW48Detector, WholeBodyDetector, RTMPoseDetector
)

# Importar procesadores
from .processors import MultiCameraProcessor

# Importar sincronización
from .synchronization import VideoSynchronizer, create_synchronizer_from_videos

# Importar ensemble
from .ensemble import EnsembleProcessor, EnsembleResult

# Importar pipeline principal
from .pipeline import VideoProcessingPipeline, video_pipeline, initialize_pipeline, get_pipeline_status

# Importar utilidades
from .utils import (
    save_keypoints_2d_frame, load_keypoints_2d_frame,
    get_available_frames, get_available_detectors,
    get_session_summary
)

# Exportar todo
__all__ = [
    # Coordinador
    'ProcessingCoordinator',
    'processing_coordinator',
    
    # Pipeline principal
    'VideoProcessingPipeline',
    'video_pipeline', 
    'initialize_pipeline',
    'process_chunk',
    
    # Datos
    'FrameResult', 
    'SyncFrameResult',
    'VideoProcessingResult',
    'MultiCameraResult',
    'ProcessingSessionResult',
    'VideoInfo',
    'SyncFrame',
    'SyncConfig',
    
    # Detectores específicos
    'BasePoseDetector',
    'VitPoseDetector',
    'HRNetW48Detector', 
    'WholeBodyDetector',
    'RTMPoseDetector',
    
    # Procesadores
    'MultiCameraProcessor',
    
    # Sincronización
    'VideoSynchronizer',
    'create_synchronizer_from_videos',
    
    # Ensemble learning
    'EnsembleProcessor',
    'EnsembleResult',
    
    # Utilidades
    'save_keypoints_2d_frame',
    'load_keypoints_2d_frame',
    'get_available_frames',
    'get_available_detectors',
    'get_session_summary'
]

# Información del módulo
__version__ = "2.0.0"
__author__ = "Gonartrosis Analysis Team"
__description__ = "Sistema simplificado para procesamiento multi-cámara con detectores MMPose específicos"

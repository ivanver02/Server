"""
Módulo de procesamiento de video reestructurado y escalable
Arquitectura modular para análisis de pose multi-cámara sincronizado
"""

# Importar coordinador principal
from .coordinator import ProcessingCoordinator, processing_coordinator

# Importar estructuras de datos
from .data import (
    KeypointResult, FrameResult, SyncFrameResult,
    VideoProcessingResult, MultiCameraResult, ProcessingSessionResult,
    VideoInfo, SyncFrame, SyncConfig
)

# Importar detectores
from .detectors import (
    BasePoseDetector, BaseDetectorManager, DetectorFactory,
    MMPoseDetector, MMPoseManager
)

# Importar procesadores
from .processors import MultiCameraProcessor

# Importar sincronización
from .synchronization import VideoSynchronizer, create_synchronizer_from_videos

# Importar ensemble
from .ensemble import EnsembleProcessor, EnsembleResult

__all__ = [
    # Coordinador principal
    'ProcessingCoordinator',
    'processing_coordinator',
    
    # Estructuras de datos
    'KeypointResult',
    'FrameResult', 
    'SyncFrameResult',
    'VideoProcessingResult',
    'MultiCameraResult',
    'ProcessingSessionResult',
    'VideoInfo',
    'SyncFrame',
    'SyncConfig',
    
    # Detectores
    'BasePoseDetector',
    'BaseDetectorManager',
    'DetectorFactory',
    'MMPoseDetector', 
    'MMPoseManager',
    
    # Procesadores
    'MultiCameraProcessor',
    
    # Sincronización
    'VideoSynchronizer',
    'create_synchronizer_from_videos',
    
    # Ensemble learning
    'EnsembleProcessor',
    'EnsembleResult'
]

# Información del módulo
__version__ = "2.0.0"
__author__ = "Processing Team"
__description__ = "Módulo escalable para procesamiento sincronizado de video multi-cámara con detección de pose"

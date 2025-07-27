"""
Estructuras de datos del módulo processing
"""
from .keypoint_result import KeypointResult, FrameResult, SyncFrameResult
from .processing_result import VideoProcessingResult, MultiCameraResult, ProcessingSessionResult
from .frame_data import VideoInfo, SyncFrame, SyncConfig

__all__ = [
    # Resultados de keypoints
    'KeypointResult',
    'FrameResult', 
    'SyncFrameResult',
    
    # Resultados de procesamiento
    'VideoProcessingResult',
    'MultiCameraResult',
    'ProcessingSessionResult',
    
    # Datos de frames y sincronización
    'VideoInfo',
    'SyncFrame',
    'SyncConfig'
]

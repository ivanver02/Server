# Inicialización del módulo de modelos
from .photo import Photo, KeypointResult
from .video import Video, VideoProcessingResult

__all__ = [
    'Photo',
    'KeypointResult', 
    'Video',
    'VideoProcessingResult'
]
# Módulo de procesamiento de video y pose detection
from .coordinator import PoseProcessingCoordinator
from .chunk_queue import ChunkProcessingQueue, ChunkProcessingTask

__all__ = [
    'PoseProcessingCoordinator',
    'ChunkProcessingQueue', 
    'ChunkProcessingTask'
]
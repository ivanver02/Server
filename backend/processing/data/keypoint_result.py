"""
Estructuras de datos para resultados de keypoints
"""
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class KeypointResult:
    """Resultado de la detección de keypoints en un frame"""
    success: bool
    keypoints: Optional[np.ndarray] = None  # (N, 2) coordenadas 2D
    scores: Optional[np.ndarray] = None     # (N,) confianza por keypoint
    bbox: Optional[np.ndarray] = None       # (4,) bounding box [x1,y1,x2,y2]
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FrameResult:
    """Resultado del procesamiento de un frame individual"""
    frame_number: int
    timestamp: float
    camera_id: int
    keypoint_results: Dict[str, KeypointResult]  # {model_name: result}
    processing_time: float = 0.0
    success: bool = True
    errors: Optional[list] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class SyncFrameResult:
    """Resultado del procesamiento de frames sincronizados"""
    frame_number: int
    timestamp: float
    camera_results: Dict[int, FrameResult]  # {camera_id: frame_result}
    available_cameras: list
    sync_quality: float = 1.0  # Calidad de sincronización (0-1)
    processing_time: float = 0.0

    def __post_init__(self):
        self.available_cameras = list(self.camera_results.keys())

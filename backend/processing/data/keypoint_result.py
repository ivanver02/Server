"""
Estructuras de datos para resultados de keypoints
"""
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class FrameResult:
    """Resultado simplificado del procesamiento de un frame individual"""
    frame_number: int
    timestamp: float
    camera_id: int
    frame_processed: bool = False  # Si se procesó el frame
    detectors_used: List[str] = None  # Detectores que funcionaron
    num_detections: int = 0  # Número de detecciones exitosas
    processing_time: float = 0.0
    success: bool = True
    errors: Optional[list] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.detectors_used is None:
            self.detectors_used = []


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

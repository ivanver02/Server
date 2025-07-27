"""
Estructuras de datos para resultados de procesamiento de video
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .keypoint_result import SyncFrameResult


@dataclass
class VideoProcessingResult:
    """Resultado del procesamiento de un video individual"""
    success: bool
    video_path: str
    camera_id: int
    total_frames: int
    processed_frames: int
    processing_time: float
    frame_results: List[Any] = None  # Lista de resultados por frame
    errors: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.frame_results is None:
            self.frame_results = []
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MultiCameraResult:
    """Resultado del procesamiento multi-cámara sincronizado"""
    success: bool
    patient_id: str
    session_id: str
    chunk_number: int
    sync_frame_results: List[SyncFrameResult]
    processing_time: float
    total_frames: int
    camera_videos: Dict[int, str]  # {camera_id: video_path}
    sync_info: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.sync_info is None:
            self.sync_info = {}
        if self.errors is None:
            self.errors = []
        self.total_frames = len(self.sync_frame_results)


@dataclass
class ProcessingSessionResult:
    """Resultado completo de una sesión de procesamiento"""
    success: bool
    patient_id: str
    session_id: str
    chunk_results: List[MultiCameraResult]
    ensemble_results: Dict[str, Any] = None
    total_processing_time: float = 0.0
    total_frames: int = 0
    keypoints_count: int = 0
    output_paths: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.ensemble_results is None:
            self.ensemble_results = {}
        if self.output_paths is None:
            self.output_paths = []
        if self.errors is None:
            self.errors = []
        
        # Calcular estadísticas
        self.total_frames = sum(chunk.total_frames for chunk in self.chunk_results)
        self.total_processing_time = sum(chunk.processing_time for chunk in self.chunk_results)

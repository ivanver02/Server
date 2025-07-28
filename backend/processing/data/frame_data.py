"""
Estructuras de datos para frames sincronizados
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    """Información de un video para sincronización"""
    camera_id: int
    video_path: Path
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int


@dataclass
class SyncFrame:
    """Frame sincronizado de múltiples cámaras"""
    timestamp: float
    frame_number: int
    camera_frames: Dict[int, np.ndarray]  # {camera_id: frame_array}
    available_cameras: List[int]
    sync_quality: float = 1.0  # Calidad de sincronización (0-1)

    def __post_init__(self):
        self.available_cameras = list(self.camera_frames.keys())


@dataclass 
class SyncConfig:
    """Configuración para sincronización de videos"""
    target_fps: Optional[float] = None  # None = usar mínimo FPS
    frame_interval: int = 1  # Procesar cada N frames
    start_time: float = 0.0  # Tiempo inicial en segundos
    end_time: Optional[float] = None  # Tiempo final (None = hasta el final)
    sync_tolerance: float = 0.1  # Tolerancia de sincronización en segundos
    quality_threshold: float = 0.8  # Umbral mínimo de calidad de sync

    def __post_init__(self):
        """Aplicar configuración centralizada si no se especifican valores"""
        try:
            from config import synchronization_config
            
            # Solo aplicar valores por defecto si no se especificaron
            if self.target_fps is None:
                self.target_fps = synchronization_config.target_fps
            if self.frame_interval == 1:  # Valor por defecto
                self.frame_interval = synchronization_config.frame_interval
            if self.sync_tolerance == 0.1:  # Valor por defecto
                self.sync_tolerance = synchronization_config.sync_tolerance
            if self.quality_threshold == 0.8:  # Valor por defecto
                self.quality_threshold = synchronization_config.quality_threshold
                
        except ImportError:
            # Si no se puede importar config, usar valores por defecto
            pass

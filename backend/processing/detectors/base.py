"""
Interfaces base para detectores de pose escalables
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import logging

from ..data import KeypointResult, FrameResult

logger = logging.getLogger(__name__)


class BasePoseDetector(ABC):
    """
    Interfaz base para todos los detectores de pose
    Permite extender a diferentes frameworks (MMPose, MediaPipe, OpenPose, etc.)
    """
    
    def __init__(self, model_name: str, model_path: Optional[Path] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializar el detector"""
        pass
    
    @abstractmethod
    def detect_frame(self, frame: np.ndarray) -> KeypointResult:
        """
        Detectar keypoints en un frame individual
        
        Args:
            frame: Frame de imagen (H, W, C)
            
        Returns:
            Resultado de la detección
        """
        pass
    
    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[KeypointResult]:
        """
        Detectar keypoints en un batch de frames
        
        Args:
            frames: Lista de frames
            
        Returns:
            Lista de resultados de detección
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Limpiar recursos del detector"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        pass
    
    def get_keypoint_names(self) -> List[str]:
        """Obtener nombres de keypoints del modelo"""
        return []
    
    def get_num_keypoints(self) -> int:
        """Obtener número de keypoints del modelo"""
        return len(self.get_keypoint_names())


class BaseDetectorManager(ABC):
    """
    Manager base para gestionar múltiples detectores
    """
    
    def __init__(self):
        self.detectors: Dict[str, BasePoseDetector] = {}
        self.active_models: List[str] = []
    
    @abstractmethod
    def register_detector(self, detector: BasePoseDetector) -> bool:
        """Registrar un nuevo detector"""
        pass
    
    @abstractmethod
    def initialize_all(self) -> bool:
        """Inicializar todos los detectores registrados"""
        pass
    
    def get_detector(self, model_name: str) -> Optional[BasePoseDetector]:
        """Obtener detector por nombre"""
        return self.detectors.get(model_name)
    
    def get_available_detectors(self) -> List[str]:
        """Obtener lista de detectores disponibles"""
        return list(self.detectors.keys())
    
    def get_active_detectors(self) -> List[str]:
        """Obtener lista de detectores activos (inicializados)"""
        return self.active_models
    
    def cleanup_all(self):
        """Limpiar todos los detectores"""
        for detector in self.detectors.values():
            try:
                detector.cleanup()
            except Exception as e:
                logger.error(f"Error limpiando detector {detector.model_name}: {e}")
        
        self.detectors.clear()
        self.active_models.clear()


class DetectorFactory:
    """
    Factory para crear detectores según el tipo
    """
    
    @staticmethod
    def create_detector(detector_type: str, model_name: str, 
                       model_path: Optional[Path] = None) -> Optional[BasePoseDetector]:
        """
        Crear detector según el tipo
        
        Args:
            detector_type: Tipo de detector ('mmpose', 'mediapipe', etc.)
            model_name: Nombre del modelo
            model_path: Ruta al modelo (opcional)
            
        Returns:
            Instancia del detector o None si no es válido
        """
        if detector_type.lower() == 'mmpose':
            from .mmpose.detector import MMPoseDetector
            return MMPoseDetector(model_name, model_path)
        # Aquí se pueden agregar otros tipos de detectores
        # elif detector_type.lower() == 'mediapipe':
        #     from .mediapipe.detector import MediaPipeDetector
        #     return MediaPipeDetector(model_name, model_path)
        
        logger.error(f"Tipo de detector no soportado: {detector_type}")
        return None

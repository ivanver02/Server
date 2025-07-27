"""
Interfaz base para detectores de pose
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BasePoseDetector(ABC):
    """
    Interfaz base para detectores de pose
    Define las operaciones básicas que debe implementar cualquier detector
    """
    
    def __init__(self, detector_name: str):
        self.detector_name = detector_name
        self.is_initialized = False
    
    @property
    def name(self) -> str:
        """Nombre del detector"""
        return self.detector_name
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Inicializar el detector con sus modelos y configuraciones
        
        Returns:
            True si se inicializó correctamente
        """
        pass
    
    @abstractmethod
    def detect_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detectar keypoints en un frame individual
        
        Args:
            frame: Frame de imagen como array numpy (H, W, C)
            
        Returns:
            Tuple (keypoints, scores) donde:
            - keypoints: Array (N, 2) con coordenadas x,y de los keypoints
            - scores: Array (N,) con confianzas [0-1] de cada keypoint
            - Ambos son None si no se detecta ninguna persona
        """
        pass
    
    def cleanup(self):
        """Limpiar recursos del detector"""
        self.is_initialized = False
        logger.info(f"Detector {self.name} limpiado")
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del detector"""
        return {
            'name': self.name,
            'is_initialized': self.is_initialized
        }

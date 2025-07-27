"""
Módulo de detectores escalable
"""
from .base import BasePoseDetector, BaseDetectorManager, DetectorFactory
from .mmpose import MMPoseDetector, MMPoseManager

__all__ = [
    # Interfaces base
    'BasePoseDetector',
    'BaseDetectorManager', 
    'DetectorFactory',
    
    # Detectores MMPose
    'MMPoseDetector',
    'MMPoseManager'
]

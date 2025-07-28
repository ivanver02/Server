"""
Detectores de pose para el sistema de análisis de gonartrosis
"""
from .base import BasePoseDetector
from .mmpose.vitpose_detector import VitPoseDetector
from .mmpose.hrnet_w48_detector import HRNetW48Detector
from .mmpose.wholebody_detector import WholeBodyDetector
from .mmpose.rtmpose_detector import RTMPoseDetector
from .mmpose.resnet50_rle_detector import ResNet50RLEDetector

__all__ = [
    'BasePoseDetector',
    'VitPoseDetector',
    'HRNetW48Detector', 
    'WholeBodyDetector',
    'RTMPoseDetector',
    'ResNet50RLEDetector'
]

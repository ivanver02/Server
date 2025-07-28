"""
Detectores MMPose específicos
"""
from .vitpose_detector import VitPoseDetector
from .hrnet_w48_detector import HRNetW48Detector
from .wholebody_detector import WholeBodyDetector
from .rtmpose_detector import RTMPoseDetector
from .resnet50_rle_detector import ResNet50RLEDetector

__all__ = [
    'VitPoseDetector',
    'HRNetW48Detector',
    'WholeBodyDetector',
    'RTMPoseDetector',
    'ResNet50RLEDetector'
]

"""
Detectores MMPose específicos
"""
from .vitpose_detector import VitPoseDetector
from .hrnet_w48_detector import HRNetW48Detector
from .wholebody_detector import WholeBodyDetector
from .rtmpose_detector import RTMPoseDetector

__all__ = [
    'VitPoseDetector',
    'HRNetW48Detector',
    'WholeBodyDetector',
    'RTMPoseDetector'
]

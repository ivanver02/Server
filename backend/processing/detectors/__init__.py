"""
Detectores de pose para análisis de marcha
"""
from .vitpose import VitPoseDetector
from .mspn import MSPNDetector
from .hrnet import HRNetDetector
from .csp import CSPDetector

__all__ = ['VitPoseDetector', 'MSPNDetector', 'HRNetDetector', 'CSPDetector']

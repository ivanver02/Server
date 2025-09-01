"""
Sistema de reconstrucción 3D para cámaras Orbbec Gemini 335Le.

Módulos:
- camera: Clase Camera para gestión de parámetros intrínsecos y extrínsecos
- calculate_extrinsics: Cálculo preciso de parámetros extrínsecos desde keypoints 2D
- triangulation: Triangulación SVD rápida y Bundle Adjustment preciso
- validation: Validación por reproyección de puntos 3D
"""

from .camera import Camera
from .calculate_extrinsics import calculate_extrinsics_from_keypoints
from .triangulation import triangulate_svd, triangulate_bundle_adjustment
from .validation import validate_reprojection

__all__ = [
    'Camera',
    'calculate_extrinsics_from_keypoints', 
    'triangulate_svd',
    'triangulate_bundle_adjustment',
    'validate_reprojection'
]

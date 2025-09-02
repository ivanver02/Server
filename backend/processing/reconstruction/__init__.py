"""
Sistema de reconstrucción 3D para análisis de marcha.

Módulos principales:
- camera: Gestión de parámetros intrínsecos y extrínsecos
- calculate_extrinsics: Cálculo de parámetros extrínsecos desde keypoints 2D
- triangulation_svd: Reconstrucción 3D rápida usando SVD
- triangulation_bundle_adjustment: Reconstrucción 3D precisa usando Bundle Adjustment
- reprojection: Validación por reproyección 2D
"""

from .camera import Camera
from .calculate_extrinsics import calculate_extrinsics_from_keypoints
from .triangulation_svd import triangulate_svd
from .triangulation_bundle_adjustment import triangulate_bundle_adjustment
from .reprojection import reproject_and_validate

__all__ = [
    'Camera',
    'calculate_extrinsics_from_keypoints',
    'triangulate_svd', 
    'triangulate_bundle_adjustment',
    'reproject_and_validate'
]

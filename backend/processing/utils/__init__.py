"""
Utilidades de procesamiento para el backend
"""
from .keypoint_io import (
    save_keypoints_2d_frame,
    load_keypoints_2d_frame,
    get_available_frames,
    get_available_detectors,
    save_frame_metadata,
    get_session_summary
)

__all__ = [
    'save_keypoints_2d_frame',
    'load_keypoints_2d_frame', 
    'get_available_frames',
    'get_available_detectors',
    'save_frame_metadata',
    'get_session_summary'
]

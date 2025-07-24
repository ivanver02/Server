# Inicialización del módulo config
from .settings import (
    server_config,
    processing_config, 
    reconstruction_config,
    data_config,
    mmpose_config
)

from .keypoint_mappings import (
    MODEL_KEYPOINT_MAPPINGS,
    GONARTROSIS_KEYPOINTS,
    COCO_SKELETON,
    get_model_keypoints,
    get_keypoint_index,
    get_gonartrosis_indices
)

from .camera_intrinsics import (
    CAMERA_INTRINSICS,
    IMAGE_RESOLUTION,
    get_camera_intrinsics,
    get_default_intrinsics,
    update_camera_intrinsics,
    get_all_camera_intrinsics
)

__all__ = [
    # Settings
    'server_config',
    'processing_config',
    'reconstruction_config', 
    'data_config',
    'mmpose_config',
    
    # Keypoint mappings
    'MODEL_KEYPOINT_MAPPINGS',
    'GONARTROSIS_KEYPOINTS',
    'COCO_SKELETON',
    'get_model_keypoints',
    'get_keypoint_index',
    'get_gonartrosis_indices',
    
    # Camera intrinsics
    'CAMERA_INTRINSICS',
    'IMAGE_RESOLUTION',
    'get_camera_intrinsics',
    'get_default_intrinsics',
    'update_camera_intrinsics',
    'get_all_camera_intrinsics'
]

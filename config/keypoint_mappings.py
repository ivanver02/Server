"""
Mapeo de keypoints para diferentes modelos de MMPose
Define qué representa cada punto en cada modelo
"""
from typing import Dict, List

# COCO 17 keypoints - Estándar para HRNet y ResNet50
COCO_17_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1  
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# COCO WholeBody 133 keypoints
COCO_WHOLEBODY_KEYPOINTS = [
    # Body (17 points - same as COCO)
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    
    # Face (68 points) - indices 17-84
    'face_0', 'face_1', 'face_2', 'face_3', 'face_4', 'face_5', 'face_6', 'face_7', 
    'face_8', 'face_9', 'face_10', 'face_11', 'face_12', 'face_13', 'face_14', 'face_15',
    'face_16', 'face_17', 'face_18', 'face_19', 'face_20', 'face_21', 'face_22', 'face_23',
    'face_24', 'face_25', 'face_26', 'face_27', 'face_28', 'face_29', 'face_30', 'face_31',
    'face_32', 'face_33', 'face_34', 'face_35', 'face_36', 'face_37', 'face_38', 'face_39',
    'face_40', 'face_41', 'face_42', 'face_43', 'face_44', 'face_45', 'face_46', 'face_47',
    'face_48', 'face_49', 'face_50', 'face_51', 'face_52', 'face_53', 'face_54', 'face_55',
    'face_56', 'face_57', 'face_58', 'face_59', 'face_60', 'face_61', 'face_62', 'face_63',
    'face_64', 'face_65', 'face_66', 'face_67',
    
    # Left hand (21 points) - indices 85-105
    'left_hand_0', 'left_hand_1', 'left_hand_2', 'left_hand_3', 'left_hand_4',
    'left_hand_5', 'left_hand_6', 'left_hand_7', 'left_hand_8', 'left_hand_9',
    'left_hand_10', 'left_hand_11', 'left_hand_12', 'left_hand_13', 'left_hand_14',
    'left_hand_15', 'left_hand_16', 'left_hand_17', 'left_hand_18', 'left_hand_19',
    'left_hand_20',
    
    # Right hand (21 points) - indices 106-126  
    'right_hand_0', 'right_hand_1', 'right_hand_2', 'right_hand_3', 'right_hand_4',
    'right_hand_5', 'right_hand_6', 'right_hand_7', 'right_hand_8', 'right_hand_9',
    'right_hand_10', 'right_hand_11', 'right_hand_12', 'right_hand_13', 'right_hand_14',
    'right_hand_15', 'right_hand_16', 'right_hand_17', 'right_hand_18', 'right_hand_19',
    'right_hand_20',
    
    # Left foot (6 points) - indices 127-132
    'left_big_toe', 'left_small_toe', 'left_heel', 'left_foot_center', 'left_foot_back', 'left_foot_front',
    
    # Right foot (6 points) - indices 133-138 (note: actually ends at 132 for 133 total)
    'right_big_toe', 'right_small_toe', 'right_heel', 'right_foot_center', 'right_foot_back', 'right_foot_front'
]

# Mapeo de modelos a sus keypoints
MODEL_KEYPOINT_MAPPINGS: Dict[str, List[str]] = {
    'hrnet_w48_coco': COCO_17_KEYPOINTS,
    'hrnet_w32_coco': COCO_17_KEYPOINTS, 
    'resnet50_rle_coco': COCO_17_KEYPOINTS,
    'wholebody_coco': COCO_WHOLEBODY_KEYPOINTS
}

# Índices de keypoints relevantes para análisis de gonartrosis
GONARTROSIS_KEYPOINTS = {
    # Keypoints principales COCO (comunes a todos los modelos)
    'coco_body': {
        'left_hip': 11,
        'right_hip': 12, 
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    },
    
    # Keypoints adicionales de pies (solo en wholebody)
    'wholebody_feet': {
        'left_big_toe': 127,
        'left_small_toe': 128,
        'left_heel': 129,
        'left_foot_center': 130,
        'left_foot_back': 131,
        'left_foot_front': 132,
        'right_big_toe': 133,
        'right_small_toe': 134, 
        'right_heel': 135,
        'right_foot_center': 136,
        'right_foot_back': 137,
        'right_foot_front': 138
    }
}

# Esqueleto para visualización (conexiones entre keypoints)
COCO_SKELETON = [
    # Cabeza
    [0, 1], [0, 2], [1, 3], [2, 4],
    # Torso
    [5, 6], [5, 11], [6, 12], [11, 12],
    # Brazo izquierdo
    [5, 7], [7, 9],
    # Brazo derecho  
    [6, 8], [8, 10],
    # Pierna izquierda
    [11, 13], [13, 15],
    # Pierna derecha
    [12, 14], [14, 16]
]

def get_model_keypoints(model_name: str) -> List[str]:
    """Obtener lista de keypoints para un modelo específico"""
    return MODEL_KEYPOINT_MAPPINGS.get(model_name, COCO_17_KEYPOINTS)

def get_keypoint_index(model_name: str, keypoint_name: str) -> int:
    """Obtener índice de un keypoint específico en un modelo"""
    keypoints = get_model_keypoints(model_name)
    try:
        return keypoints.index(keypoint_name)
    except ValueError:
        return -1

def get_gonartrosis_indices(model_name: str) -> Dict[str, int]:
    """Obtener índices de keypoints relevantes para gonartrosis"""
    indices = {}
    
    # Keypoints básicos COCO (disponibles en todos los modelos)
    for keypoint, idx in GONARTROSIS_KEYPOINTS['coco_body'].items():
        indices[keypoint] = idx
    
    # Keypoints adicionales solo para wholebody
    if 'wholebody' in model_name:
        for keypoint, idx in GONARTROSIS_KEYPOINTS['wholebody_feet'].items():
            indices[keypoint] = idx
            
    return indices

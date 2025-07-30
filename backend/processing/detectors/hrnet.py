"""
Detector HRNet para análisis de pose
"""
from .base import BasePoseDetector


class HRNetDetector(BasePoseDetector):
    """
    Detector de pose utilizando HRNet (High-Resolution Network)
    """
    
    def __init__(self):
        super().__init__(model_name="hrnet", config_key="hrnet")
        # HRNet usa WholeBody keypoints (133 keypoints total)
        # 17 body + 6 feet + 42 hands + 68 face = 133
        self.keypoints_names = [
            # Body keypoints (17) - COCO format
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
            
            # Foot keypoints (6)
            "left_big_toe", "left_small_toe", "left_heel",
            "right_big_toe", "right_small_toe", "right_heel",
            
            # Face keypoints (68) - following 68-point facial landmark convention
            "face_0", "face_1", "face_2", "face_3", "face_4", "face_5", "face_6", "face_7", "face_8",
            "face_9", "face_10", "face_11", "face_12", "face_13", "face_14", "face_15", "face_16",
            "face_17", "face_18", "face_19", "face_20", "face_21", "face_22", "face_23", "face_24",
            "face_25", "face_26", "face_27", "face_28", "face_29", "face_30", "face_31", "face_32",
            "face_33", "face_34", "face_35", "face_36", "face_37", "face_38", "face_39", "face_40",
            "face_41", "face_42", "face_43", "face_44", "face_45", "face_46", "face_47", "face_48",
            "face_49", "face_50", "face_51", "face_52", "face_53", "face_54", "face_55", "face_56",
            "face_57", "face_58", "face_59", "face_60", "face_61", "face_62", "face_63", "face_64",
            "face_65", "face_66", "face_67",
            
            # Left hand keypoints (21)
            "left_thumb_1", "left_thumb_2", "left_thumb_3", "left_thumb_4",
            "left_forefinger_1", "left_forefinger_2", "left_forefinger_3", "left_forefinger_4",
            "left_middle_finger_1", "left_middle_finger_2", "left_middle_finger_3", "left_middle_finger_4",
            "left_ring_finger_1", "left_ring_finger_2", "left_ring_finger_3", "left_ring_finger_4",
            "left_pinky_finger_1", "left_pinky_finger_2", "left_pinky_finger_3", "left_pinky_finger_4",
            "left_hand_root",
            
            # Right hand keypoints (21)
            "right_thumb_1", "right_thumb_2", "right_thumb_3", "right_thumb_4",
            "right_forefinger_1", "right_forefinger_2", "right_forefinger_3", "right_forefinger_4",
            "right_middle_finger_1", "right_middle_finger_2", "right_middle_finger_3", "right_middle_finger_4",
            "right_ring_finger_1", "right_ring_finger_2", "right_ring_finger_3", "right_ring_finger_4",
            "right_pinky_finger_1", "right_pinky_finger_2", "right_pinky_finger_3", "right_pinky_finger_4",
            "right_hand_root"
        ]

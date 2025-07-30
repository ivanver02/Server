"""
Detector MSPN para análisis de pose
"""
from .base import BasePoseDetector


class MSPNDetector(BasePoseDetector):
    """
    Detector de pose utilizando MSPN (Multi-Stage Pose Network)
    """
    
    def __init__(self):
        super().__init__(model_name="mspn", config_key="mspn")
        # MSPN usa keypoints COCO (17 keypoints)
        self.keypoints_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

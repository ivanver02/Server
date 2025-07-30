"""
Detector VitPose para análisis de pose
"""
from .base import BasePoseDetector


class VitPoseDetector(BasePoseDetector):
    """
    Detector de pose utilizando VitPose
    """
    
    def __init__(self):
        super().__init__(model_name="vitpose", config_key="vitpose")
        # VitPose usa keypoints COCO (17 keypoints)
        self.keypoints_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

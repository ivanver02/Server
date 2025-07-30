from mmpose.apis import MMPoseInferencer
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint

# Lista de nombres de keypoints COCO
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 1. Inicializar el inferenciador
inferencer = MMPoseInferencer(
    pose2d='/home/work/Server/mmpose_models/configs/pose2d/td-hm_4xmspn50_8xb32-210e_coco-256x192.py',
    pose2d_weights='/home/work/Server/mmpose_models/checkpoints/4xmspn50_coco_256x192-7b837afb_20201123.pth',
    device='cuda:0' if True else 'cpu'                      # Usar GPU si está disponible
)

input_video = '0.mp4'

# 3. Ejecutas la inferencia sobre el vídeo
result_generator = inferencer(
    input_video,
    # Para que devuelva imágenes con keypoints dibujados en cada frame
    return_vis=True,
    # Carpeta donde se guardarán los frames visualizados
    vis_out_dir='resultados_video',
    # Carpeta donde se volcarán los JSON de predicciones
    pred_out_dir='predicciones_video',
    # Si quieres además recrear un vídeo de salida con los keypoints
    video_out='resultados_video/atleta_out.mp4',
    # Parámetros de dibujo
    radius=4,
    thickness=2,
    skeleton_style='mmpose',
    show=False
)


# 4. Itera a través de los resultados (uno por frame)
for frame_idx, results in enumerate(result_generator):
    print(f"\nFrame {frame_idx}:",
          np.array(results['predictions'][0][0]['keypoints']))
    print(f"\nConfianzas: {results['predictions'][0][0]['keypoint_scores']}" )

# print(f"\n KEYPOINTS: {results['predictions'][0][0]['keypoints']}")
# print(f"\n CONFIDENCE: {results['predictions'][0][0]['keypoint_scores']}")
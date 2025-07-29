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
    pose2d='/home/work/Server/mmpose_models/configs/pose2d/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py',
    pose2d_weights='/home/work/Server/mmpose_models/checkpoints/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth',
    device='cuda:0' if True else 'cpu'                      # Usar GPU si está disponible
)

# 2. Especificar ruta de la imagen de entrada
input_image = 'atleta.jpg'  # Cambia esto por tu ruta de imagen

# 3. Realizar inferencia y obtener resultados
result_generator = inferencer(
    input_image,
    return_vis=True,            # Devolver imagen visualizada
    vis_out_dir='resultados',   # Directorio para guardar salidas
    pred_out_dir='predicciones',# Directorio para archivos JSON
    radius=5,                   # Tamaño de los puntos clave
    thickness=2,                # Grosor de las líneas
    skeleton_style='mmpose',    # Estilo de conexiones
    draw_heatmap=False,         # No dibujar heatmap
    show=False                  # No mostrar ventana automáticamente
)

# 4. Procesar resultados
results = next(result_generator)

# print(f"\n KEYPOINTS: {results['predictions'][0][0]['keypoints']}")
# print(f"\n CONFIDENCE: {results['predictions'][0][0]['keypoint_scores']}")

img_bgr = cv2.imread(input_image)  # Asegúrate de cambiar esta ruta si es diferente
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

KEYPOINTS = results['predictions'][0][0]['keypoints']

xs = [kp[0] for kp in KEYPOINTS]
ys = [kp[1] for kp in KEYPOINTS]

# 4) Dibuja
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.scatter(xs, ys,
            cmap='jet',        # paleta de colores
            alpha=0.8,         # algo de transparencia
           )


# 6) Guardar imagen con nombre especificado
output_path = 'atleta_vitpose.jpg'
plt.axis('off')
plt.title('Keypoints sobre la imagen')
plt.tight_layout()

# Guardar la imagen en el directorio actual
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
print(f"Imagen guardada como {output_path}")
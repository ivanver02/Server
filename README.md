# Sistema de Análisis de Marcha para Detección de Gonartrosis

Este proyecto es el backend para el procesamiento de video y reconstrucción 3D de keypoints, desarrollado por la Universidad de Málaga y el Hospital Costa del Sol. El sistema está diseñado para analizar la marcha humana y detectar patrones relacionados con la gonartrosis, empleando procesamiento multi-cámara, modelos de pose 2D y triangulación 3D.

Este proyecto (Server) está diseñado para funcionar conjuntamente con el repositorio Client, que gestiona la captura, grabación y envío de video multi-cámara. Server se encarga del procesamiento avanzado de los videos, detección de keypoints y reconstrucción 3D. Ambos forman el flujo completo de análisis de marcha, permitiendo una integración clínica e investigadora robusta. Para el funcionamiento completo, consulta y utiliza ambos repositorios.

---
## Descripción del proyecto

El servidor recibe chunks de video desde varias cámaras, extrae los frames, detecta keypoints 2D usando varios modelos de MMPose, realiza un ensemble ponderado de los resultados y reconstruye los keypoints en 3D. El sistema está pensado para funcionar en entornos clínicos y de investigación, permitiendo la evaluación triplanar de la rodilla y la obtención de métricas precisas para el diagnóstico.

---
## Modelos empleados


El sistema utiliza los siguientes modelos de MMPose:
 - VitPose (COCO, 17 keypoints)
 - HRNet (WholeBody, 133 keypoints: cuerpo, pies, manos, cara)
 - CSP (WholeBody, 133 keypoints: cuerpo, pies, manos, cara)

Para una descripción detallada de las clases y métodos principales del backend, consulta el archivo [`docs/main_classes.md`](docs/main_classes.md).

Cada modelo se integra como un detector independiente, con sus propias ponderaciones y lista de keypoints. Es posible añadir nuevos modelos de MMPose creando una clase que herede de [`backend/processing/detectors/base.py`](backend/processing/detectors/base.py), o integrar modelos externos sobreescribiendo los métodos necesarios.

<div style="background-color:#e3f2fd; border-left:6px solid #1976d2; padding:10px; margin-bottom:10px;">
Los archivos de <strong>checkpoint</strong> necesarios para los modelos se encuentran en una <strong>release</strong> del proyecto. Deben guardarse en la ruta exacta [`Server/mmpose_models/checkpoints/`](mmpose_models/checkpoints/).
</div>

---
## Estructura de archivos del proyecto

```
Server/
├── app.py
├── main.py
├── config/
│   ├── settings.py
│   ├── __init__.py
│   └── camera_intrinsics.py
├── backend/
│   ├── processing/
│   │   ├── ensemble/
│   │   │   └── ensemble_processor.py
│   │   ├── detectors/
│   │   │   ├── base.py
│   │   │   ├── vitpose.py
│   │   │   ├── mspn.py
│   │   │   ├── hrnet.py
│   │   │   ├── csp.py
│   │   │   └── __init__.py
│   │   ├── reconstruction/
│   │   │   ├── camera.py
│   │   │   ├── calculate_extrinsics.py
│   │   │   ├── triangulation_svd.py
│   │   │   ├── bundle_adjustment.py
│   │   │   ├── perform_reconstruction.py
│   │   │   ├── reprojection.py
│   │   │   ├── analyze_3D_keypoint.py
│   │   │   ├── complete_analysis.py
│   │   │   └── __init__.py
│   │   ├── coordinator.py
│   │   └── __init__.py
│   ├── tests/
│   │   ├── csp.py
│   │   ├── hrnet_w48_wholebody.py
│   │   ├── mspn.py
│   │   ├── vitpose.py
│   │   ├── video.py
│   │   ├── reconstruccion_2D.py
│   │   └── __init__.py
│   └── __init__.py
├── mmpose_models/
│   ├── configs/
│   │   ├── pose2d/
│   │   │   ├── td-hm_ViTPose-large_8xb64-210e_coco-256x192.py
│   │   │   ├── td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py
│   │   │   ├── td-hm_4xmspn50_8xb32-210e_coco-256x192.py
│   │   │   └── cspnext-m_udp_8xb64-210e_coco-wholebody-256x192.py
│   │   └── default_runtime.py
│   └── checkpoints/
├── data/
│   ├── unprocessed/
│   │   └── <paciente>/
│   │       └── <sesion>/
│   │           └── <camara>/
│   │               └── chunks/
│   │                   └── chunk_<id>.mp4
│   │               └── keypoints2D/
│   │                   └── <detector>/
│   │                       └── <camara>/
│   │                           ├── coordinates.npy
│   │                           └── confidence.npy
│   ├── processed/
│   │   ├── 2D_keypoints/
│   │   │   └── <paciente>/<sesion>/<camara>/
│   │   │       ├── coordinates/
│   │   │       │   └── {frame_id}_{chunk_id}.npy
│   │   │       └── confidence/
│   │   │           └── {frame_id}_{chunk_id}.npy
│   │   ├── 3D_keypoints/
│   │   │   └── <paciente>/<sesion>/{frame_id}_{chunk_id}.npy
│   │   ├── annotated_videos/
│   │   │   └── <paciente>/<sesion>/<camara>/<detector>/video_annotated.mp4
│   │   └── photos_from_video/
│   │       └── <paciente>/<sesion>/<camara>/frames/
│   ├── logs/
│   └── ...
├── docs/
│   └── main_classes.md
├── .gitignore
├── LICENSE.md
├── README.md
└── requirements.txt
```


---
## Cómo ejecutar el servidor


<div style="background-color:#e3f2fd; border-left:6px solid #1976d2; padding:10px; margin-bottom:10px;">
Instala las dependencias antes de ejecutar el servidor.
</div>

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```
2. Configura los parámetros en [`config/settings.py`](config/settings.py) según tu entorno (puerto, GPUs, rutas de modelos, etc.).
3. Ejecuta el servidor:
```bash
python main.py
```
<div style="background-color:#fffde7; border-left:6px solid #fbc02d; padding:10px; margin-bottom:10px;">
El servidor se iniciará en el puerto configurado (por defecto 5000). Asegúrate de que el puerto esté abierto y accesible. Si usas el sistema con el repositorio Client, verifica que la configuración del servidor en <code>Client/backend/config/settings.py</code> coincida con la IP y puerto del servidor.
</div>


---
## API Endpoints


- `POST /api/session/start`: Inicializa una nueva sesión de grabación y procesamiento.
- `POST /api/chunks/receive`: Recibe chunks de video desde el cliente para procesar.
- `POST /api/session/end`: Finaliza la sesión de grabación, calcula el chunk máximo y permite continuar el procesamiento.
- `POST /api/session/cancel`: Cancela la sesión y elimina todos los datos procesados.
- `GET /api/session/status`: Consulta el estado actual de la sesión.
- `GET /health`: Verifica el estado del servidor.
- `POST /api/cameras/recalibrate`: Recalibra los parámetros extrínsecos de las cámaras.


---
## Consideraciones importantes

- Se pueden generar demos visuales en [`data/processed/annotated_videos`](data/processed/annotated_videos) para ver la reconstrucción de keypoints 2D por modelo y cámara. Para ello, activa `save_annotated_videos = True` en [`config/settings.py`](config/settings.py) y limita `available_gpus` a una sola GPU.

- Solo puede haber una sesión de grabación activa, pero pueden procesarse varias sesiones simultáneamente.

- El archivo [`backend/tests/reconstruccion_2D.py`](backend/tests/reconstruccion_2D.py) se puede emplear para visualizar la reconstrucción 2D del flujo completo, para estudiar si funciona correctamente.

<div style="background-color:#ffebee; border-left:6px solid #d32f2f; padding:10px; margin-bottom:10px;">
<strong>Advertencia:</strong> Si alguna cámara falla, es necesario reiniciar los dos servidores Flask y el switch de las cámaras.
</div>


---
## Pipeline completo del proyecto

- Al comenzar la grabación, se ejecuta `/api/session/start` para inicializar todo lo necesario.
- Durante la grabación, el cliente envía chunks de video al servidor.
- La sesión puede finalizarse o cancelarse:
  - Si se cancela, se ejecuta `/api/session/cancel`, se eliminan los datos y se finaliza la sesión.
  - Si se finaliza, se ejecuta `/api/session/end`, se determina el chunk máximo y se finaliza la sesión de grabación, aunque el procesamiento puede continuar.


---
## Detectors

- Todos los detectores heredan de [`backend/processing/detectors/base.py`](backend/processing/detectors/base.py), donde se define la inicialización, manejo de GPU, guardado de vídeos anotados y procesamiento de chunks comunes.
- Las características específicas de cada detector (ponderaciones, keypoints, etc.) se definen en cada clase concreta.
- Para añadir un detector de MMPose, basta con crear una clase que herede de `BasePoseDetector`.
- Para integrar otros detectores, se pueden sobreescribir los métodos `initialize` y `process_chunk`, o crear una clase con los métodos necesarios adaptados al nuevo modelo.


---
## Pose Processing Coordinator

- El coordinador abstrae el uso de varios detectores de pose 2D y permite escalar el sistema con diferentes modelos.
- Gestiona la alternancia y asignación de GPUs.
- Al recibir el primer chunk, inicializa los detectores seleccionados.
- Para cada chunk recibido, ejecuta `process_chunk` en todos los detectores activos.


---
## Ensembling

- Los detectores que participan en el ensemble se indican en `detector_instances`.
- Cada detector define `ensemble_confidence_weights`, que asigna una ponderación a cada keypoint.
- Los keypoints finales se calculan ponderando linealmente los resultados de cada modelo, considerando tanto la confianza del detector como la ponderación asignada.
- La confianza final de cada keypoint se calcula también de forma ponderada.


### Flujo de trabajo
- Al iniciar la grabación, se registra la sesión con `register_session_start`.
- Al finalizar, se calcula el máximo id de chunk con `get_max_chunk`.
- Al procesar un chunk, se ejecuta `ensemble_processor.register_chunk_completion`, que inicia el ensembling de forma asíncrona cuando todas las cámaras han procesado el último chunk.
- El ensembling se realiza con `process_session_ensemble`, que delega el procesamiento de cada chunk y cámara a `_process_chunk_ensemble`.
- `_get_all_frame_files` obtiene los datos de los detectores en la estructura adecuada.
- Para cada frame, `_combine_keypoints` realiza la combinación ponderada.
- Los resultados se guardan con `_save_single_frame_result`.


---
## Reconstrucción 3D

El sistema implementa un pipeline completo de reconstrucción 3D de keypoints del que podemos destacar los siguientes apartados:

### **Flujo de Procesamiento**

1. **Estimación de parámetros extrínsecos**: Se comienza estimando los parámetros extrínsecos de cada cámara usando correspondencias de keypoints 2D entre múltiples vistas, estableciendo la geometría espacial del sistema multi-cámara.

2. **Estimación de la reconstrucción 3D**: A partir de los extrínsecos iniciales, se realiza triangulación 3D de los keypoints 2D detectados, obteniendo las coordenadas espaciales de cada punto anatómico.

3. **Optimización conjunta**: Se optimizan estas estimaciones conjuntamente mediante bundle adjustment, buscando reducir el error de reproyección y refinando tanto los parámetros de las cámaras como las posiciones 3D de los keypoints.

### **Archivos del Sistema**

#### **Flujo Principal (Integrados)**
- [`camera.py`](backend/processing/reconstruction/camera.py): Gestiona diversas interacciones que se realizan con la cámara, incluyendo matrices de proyección y transformaciones.
- [`calculate_extrinsics.py`](backend/processing/reconstruction/calculate_extrinsics.py): Realiza la primera estimación de los parámetros extrínsecos usando geometría epipolar y correspondencias entre vistas.
- [`triangulation_svd.py`](backend/processing/reconstruction/triangulation_svd.py): Implementa triangulación mediante SVD para obtener la primera reconstrucción 3D a partir de los keypoints 2D.
- [`bundle_adjustment.py`](backend/processing/reconstruction/bundle_adjustment.py): Emplea, a partir de una estimación inicial de extrínsecos y reconstrucción 3D, un método iterativo para reducir el error de reproyección mediante optimización no lineal.
- [`perform_reconstruction.py`](backend/processing/reconstruction/perform_reconstruction.py): Orquesta todos los pasos del pipeline de reconstrucción 3D para que sea fácilmente integrable con el flujo principal del sistema.

#### **Herramientas de Análisis (Independientes)**
- [`reprojection.py`](backend/processing/reconstruction/reprojection.py): Calcula el error de reproyección para evaluar la calidad de la reconstrucción 3D y los parámetros de las cámaras.
- [`analyze_3D_keypoint.py`](backend/processing/reconstruction/analyze_3D_keypoint.py): Muestra las coordenadas 3D de cada keypoint detectado y proporciona estimaciones detalladas de medidas corporales.
- [`complete_analysis.py`](backend/processing/reconstruction/complete_analysis.py): Herramienta de análisis completo que muestra, para un paciente, sesión, chunk y frame especificados:
  - Parámetros extrínsecos de la estimación inicial y después de la optimización
  - Reconstrucciones 3D tanto de la triangulación inicial como de la versión optimizada
  - Ángulos de flexión de ambas rodillas para análisis biomecánico
  - Estimaciones de medidas corporales para cada método de reconstrucción
  - Análisis de keypoints 2D para cada cámara individual
  - Errores de reproyección para cada cámara y método

### **Características Técnicas**

- **Calibración robusta**: Primera estimación de extrínsecos usando múltiples frames para mayor precisión
- **Triangulación SVD**: Método matemáticamente robusto para la reconstrucción 3D inicial
- **Bundle Adjustment**: Optimización iterativa que minimiza el error de reproyección global
- **Escalado anatómico**: Normalización basada en medidas antropométricas (altura nariz-tobillo)
- **Análisis biomecánico**: Cálculo automático de ángulos articulares y medidas corporales

<div style="background-color:#e8f5e8; border-left:6px solid #4caf50; padding:10px; margin-bottom:10px;">
<strong>Integración:</strong> La reconstrucción 3D se ejecuta automáticamente después del ensemble, almacenando los resultados en <code>data/processed/3D_keypoints/</code> con formato <code>{frame_id}_{chunk_id}.npy</code>.
</div>


---
## Configuraciones

- Toda la configuración está centralizada en la carpeta [`config/`](config/).
- El archivo principal es [`config/settings.py`](config/settings.py), donde se definen rutas, GPUs, parámetros de procesamiento, etc.


---
## Testing

<div style="background-color:#fffde7; border-left:6px solid #fbc02d; padding:10px; margin-bottom:10px;">
<strong>Consejo:</strong> Utiliza la carpeta <code>backend/tests/</code> para prototipos y pruebas manuales antes de integrar cambios en el sistema principal.
</div>

La carpeta [`backend/tests/`](backend/tests/) no está pensada para pruebas automáticas, sino como espacio para desarrollar código aislado que posteriormente se integra en el proyecto principal.


---
## Licencia

Este proyecto está licenciado bajo Apache License 2.0. Los modelos y configuraciones de MMPose también están bajo Apache 2.0. Consulta el archivo [`LICENSE.md`](LICENSE.md) para más detalles, incluyendo la cita académica recomendada para MMPose.


---
Desarrollado por la Universidad de Málaga y el Hospital Costa del Sol.

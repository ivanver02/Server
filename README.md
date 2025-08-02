
# Sistema de Análisis de Marcha para Detección de Gonartrosis

Este proyecto es el backend para el procesamiento de video y reconstrucción 3D de keypoints, desarrollado por la Universidad de Málaga y el Hospital Costa del Sol. El sistema está diseñado para analizar la marcha humana y detectar patrones relacionados con la gonartrosis, empleando procesamiento multi-cámara, modelos de pose 2D y triangulación 3D.


---
## Descripción del proyecto

El servidor recibe chunks de video desde varias cámaras, extrae los frames, detecta keypoints 2D usando varios modelos de MMPose, realiza un ensemble ponderado de los resultados y reconstruye los keypoints en 3D. El sistema está pensado para funcionar en entornos clínicos y de investigación, permitiendo la evaluación triplanar de la rodilla y la obtención de métricas precisas para el diagnóstico.


---
## Modelos empleados


El sistema utiliza los siguientes modelos de MMPose:
 - VitPose (COCO, 17 keypoints)
 - HRNet (WholeBody, 133 keypoints: cuerpo, pies, manos, cara)
 - CSP (WholeBody, 133 keypoints: cuerpo, pies, manos, cara)

Cada modelo se integra como un detector independiente, con sus propias ponderaciones y lista de keypoints. Es posible añadir nuevos modelos de MMPose creando una clase que herede de `BasePoseDetector`, o integrar modelos externos sobreescribiendo los métodos necesarios.

<div style="background-color:#e3f2fd; border-left:6px solid #1976d2; padding:10px; margin-bottom:10px;">
Los archivos de <strong>checkpoint</strong> necesarios para los modelos se encuentran en una <strong>release</strong> del proyecto. Deben guardarse en la ruta exacta <code>Server/mmpose_models/checkpoints/</code>.
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
│   │   │   └── ...
│   │   ├── coordinator.py
│   │   └── ...
│   └── ...
├── mmpose_models/
│   ├── configs/
│   ├── checkpoints/
│   └── ...
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
│   │   │   └── <paciente>/<sesion>/<camara>/<detector>/keypoints2D.npy
│   │   ├── 3D_keypoints/
│   │   │   └── <paciente>/<sesion>/keypoints3D.npy
│   │   ├── annotated_videos/
│   │   │   └── <paciente>/<sesion>/<camara>/<detector>/video_annotated.mp4
│   │   └── photos_from_video/
│   │       └── <paciente>/<sesion>/<camara>/frames/
│   └── ...
├── LICENSE.md
└── README.md
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
2. Configura los parámetros en `config/settings.py` según tu entorno (puerto, GPUs, rutas de modelos, etc.).
3. Ejecuta el servidor:
```bash
python main.py
```
<div style="background-color:#fffde7; border-left:6px solid #fbc02d; padding:10px; margin-bottom:10px;">
El servidor se iniciará en el puerto configurado (por defecto 5000). Asegúrate de que el puerto esté abierto y accesible.
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

- Se pueden generar demos visuales en <code>Server/data/processed/annotated_videos</code> para ver la reconstrucción de keypoints 2D por modelo y cámara. Para ello, activa <code>save_annotated_videos = True</code> en <code>settings.py</code> y limita <code>available_gpus</code> a una sola GPU.

- Solo puede haber una sesión de grabación activa, pero pueden procesarse varias sesiones simultáneamente.

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

- Todos los detectores heredan de `base.py`, donde se define la inicialización, manejo de GPU, guardado de vídeos anotados y procesamiento de chunks comunes.
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
- Al finalizar, se calcula el máximo id de chunk con `register_session_end`.
- Al procesar un chunk, se ejecuta `ensemble_processor.register_chunk_completion`, que inicia el ensembling de forma asíncrona cuando todas las cámaras han procesado el último chunk.
- El ensembling se realiza con `process_session_ensemble`, que delega el procesamiento de cada chunk y cámara a `_process_chunk_ensemble`.
- `_get_all_frame_files` obtiene los datos de los detectores en la estructura adecuada.
- Para cada frame, `_combine_keypoints` realiza la combinación ponderada.
- Los resultados se guardan con `_save_single_frame_result`.


---
## Configuraciones

- Toda la configuración está centralizada en la carpeta `config/`.
- El archivo principal es `settings.py`, donde se definen rutas, GPUs, parámetros de procesamiento, etc.


---
## Testing

<div style="background-color:#fffde7; border-left:6px solid #fbc02d; padding:10px; margin-bottom:10px;">
<strong>Consejo:</strong> Utiliza la carpeta <code>testing</code> para prototipos y pruebas manuales antes de integrar cambios en el sistema principal.
</div>

La carpeta `testing` no está pensada para pruebas automáticas, sino como espacio para desarrollar código aislado que posteriormente se integra en el proyecto principal.


---
## Licencia

Este proyecto está licenciado bajo Apache License 2.0. Los modelos y configuraciones de MMPose también están bajo Apache 2.0. Consulta el archivo `LICENSE.md` para más detalles, incluyendo la cita académica recomendada para MMPose.


---
## Créditos

Desarrollado por la Universidad de Málaga y el Hospital Costa del Sol.

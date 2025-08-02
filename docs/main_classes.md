# Documentación de Clases Principales

Este documento describe las clases principales del backend del sistema de análisis de marcha, incluyendo sus métodos y atributos más relevantes. Se cubren las clases de los archivos `base.py`, `ensemble_processor.py`, `coordinator.py` y el detector concreto `vitpose.py`.

---

## 1. `BasePoseDetector` (`base.py`)

Clase base para todos los detectores de pose 2D. Proporciona la estructura común y métodos esenciales para la inicialización, procesamiento y gestión de resultados.

**Atributos principales:**
- `model_name`: Nombre del modelo asociado al detector.
- `config_key`: Clave de configuración para el detector.
- `keypoints_names`: Lista de nombres de los keypoints que detecta el modelo.
- `ensemble_confidence_weights`: Ponderaciones para el ensembling de keypoints.

**Métodos principales:**
- `__init__()`: Inicializa el detector con nombre y configuración.
- `initialize()`: Carga el modelo y sus parámetros.
- `process_chunk(chunk_path)`: Procesa un chunk de vídeo y extrae los keypoints.
- `save_annotated_video()`: Guarda el vídeo con anotaciones de keypoints.

---

## 2. `EnsembleProcessor` (`ensemble_processor.py`)

Gestiona la combinación de resultados de varios detectores para obtener keypoints finales más robustos.

**Atributos principales:**
- `detector_instances`: Lista de instancias de detectores activos.
- `ensemble_confidence_weights`: Ponderaciones para cada detector y keypoint.
- `session_chunks`: Estructura para gestionar los chunks por sesión y cámara.

**Métodos principales:**
- `register_session_start()`: Inicializa la estructura de sesión.
- `register_chunk_completion()`: Marca un chunk como procesado y lanza el ensembling si corresponde.
- `process_session_ensemble()`: Ejecuta el ensembling sobre todos los chunks y cámaras.
- `_combine_keypoints()`: Realiza la combinación ponderada de keypoints.
- `_save_single_frame_result()`: Guarda el resultado final de keypoints para un frame.

---

## 3. `PoseProcessingCoordinator` (`coordinator.py`)

Coordina el uso de múltiples detectores y la asignación de recursos (GPUs) para el procesamiento eficiente de los chunks.

**Atributos principales:**
- `detectors`: Lista de detectores activos.
- `available_gpus`: GPUs disponibles para el procesamiento.
- `session_info`: Información de la sesión actual.

**Métodos principales:**
- `initialize_detectors()`: Inicializa los detectores seleccionados.
- `assign_gpus()`: Asigna GPUs a los detectores.
- `process_chunk(chunk_path)`: Procesa un chunk usando todos los detectores activos.
- `get_status()`: Devuelve el estado actual del procesamiento.

---

## 4. `VitPoseDetector` (`vitpose.py`)

Ejemplo de detector concreto basado en el modelo VitPose. Hereda de `BasePoseDetector` y define los parámetros específicos para este modelo.

**Atributos principales:**
- `model_name`: "vitpose"
- `config_key`: "vitpose"
- `keypoints_names`: Lista de 17 keypoints (COCO)
- `ensemble_confidence_weights`: Ponderaciones específicas para VitPose

**Métodos principales:**
- `__init__()`: Inicializa el detector con los parámetros de VitPose.
- `initialize()`: Carga el modelo VitPose y sus configuraciones.
- `process_chunk(chunk_path)`: Procesa el chunk y extrae los keypoints usando VitPose.

---

Cada clase está diseñada para ser extensible y facilitar la integración de nuevos modelos o el ajuste de la lógica de procesamiento. Para más detalles, consulta el código fuente correspondiente en cada archivo.

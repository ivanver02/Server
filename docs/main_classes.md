# Documentación de Clases Principales

Este documento describe las clases principales del backend del sistema de análisis de marcha, incluyendo sus métodos y atributos más relevantes. Se cubren las clases de los archivos `base.py`, `ensemble_processor.py`, `coordinator.py` y el detector concreto `vitpose.py`.

---

## 1. `BasePoseDetector` (`base.py`)

Clase base para todos los detectores de pose 2D. Proporciona la estructura común y métodos esenciales para la inicialización, procesamiento y gestión de resultados.

**Atributos principales:**
- `model_name`: Nombre del modelo asociado al detector.
- `config_key`: Clave de configuración para el detector.
- `keypoints_names`: Lista de nombres de los keypoints que detecta el modelo.
- `ensemble_confidence_weights`: Ponderaciones para el ensembling de keypoints. Se encuentra en cada detector en específico.

**Métodos principales:**
- `__init__()`: Inicializa el detector con nombre y configuración.
- `initialize()`: Carga el modelo y sus parámetros.
- `process_chunk(chunk_path)`: Procesa un chunk de vídeo y extrae los keypoints.

---

## 2. `EnsembleProcessor` (`ensemble_processor.py`)

Gestiona la combinación de resultados de varios detectores para obtener keypoints finales más robustos.

**Atributos principales:**
- `detector_instances`: Lista de instancias de detectores activos.
- `final_keypoint_names`: Lista de nombres de keypoints resultante tras considerar todos los detectores con sus ponderaciones de keypoints.

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

**Métodos principales:**
- `initialize_all()`: Inicializa los detectores seleccionados.
- `process_chunk(chunk_path)`: Procesa un chunk usando todos los detectores activos.

---

## 4. `VitPoseDetector` (`vitpose.py`)

Ejemplo de detector concreto basado en el modelo VitPose. Hereda de `BasePoseDetector` y define los parámetros específicos para este modelo.

**Atributos principales:**
- `model_name`: "vitpose"
- `config_key`: "vitpose"
- `keypoints_names`: Lista de 17 keypoints (COCO)
- `ensemble_confidence_weights`: Ponderaciones específicas para VitPose

**Métodos principales heredados de `BasePoseDetector`**

---

## 5. Reconstrucción 3D (`reconstruction/`)

El sistema de reconstrucción 3D implementa un pipeline completo para convertir keypoints 2D detectados por múltiples cámaras en coordenadas tridimensionales precisas. Este módulo incluye clases y funciones especializadas para cada etapa del proceso.

### 5.1. `Camera` (`camera.py`)

Clase que encapsula los parámetros intrínsecos y extrínsecos de cada cámara del sistema multi-cámara.

**Atributos principales:**
- `camera_id`: Identificador de la cámara (e.g., "camera0", "camera1", "camera2").
- `K`: Matriz de parámetros intrínsecos 3x3.
- `dist_coeffs`: Coeficientes de distorsión de la lente.
- `R`: Matriz de rotación 3x3 (extrínsecos).
- `t`: Vector de traslación 3x1 (extrínsecos).

**Métodos principales:**
- `create(camera_id)`: Método estático que crea una instancia con parámetros intrínsecos desde configuración.
- `P`: Propiedad que devuelve la matriz de proyección 3x4 = K [R|t].
- `project(points_3d)`: Proyecta puntos 3D a coordenadas 2D en píxeles.

### 5.2. Estimación de Extrínsecos (`calculate_extrinsics.py`)

Módulo que implementa algoritmos para estimar los parámetros extrínsecos de las cámaras usando correspondencias de keypoints 2D.

**Funciones principales:**
- `estimate_fundamental_matrix_ransac()`: Estima la matriz fundamental usando RANSAC para robustez.
- `compute_fundamental_8point()`: Implementa el método de 8 puntos para calcular la matriz fundamental.
- `essential_from_fundamental()`: Convierte matriz fundamental a esencial usando parámetros intrínsecos.
- `decompose_essential_matrix()`: Descompone la matriz esencial en rotación y traslación.
- `estimate_extrinsics()`: Función principal que coordina todo el proceso de estimación.

### 5.3. Triangulación SVD (`triangulation_svd.py`)

Implementa triangulación 3D robusta mediante descomposición en valores singulares (SVD).

**Funciones principales:**
- `triangulate_frame_svd()`: Triangula todos los keypoints de un frame usando múltiples vistas.
- Utiliza el método DLT (Direct Linear Transform) con SVD para mayor estabilidad numérica.
- Maneja automáticamente casos con keypoints faltantes o de baja confianza.

### 5.4. Bundle Adjustment (`bundle_adjustment.py`)

Optimización no lineal que refina conjuntamente los parámetros de las cámaras y las posiciones 3D de los keypoints.

**Funciones principales:**
- `bundle_adjustment()`: Función principal de optimización que minimiza el error de reproyección.
- `rodrigues_to_rotation_matrix()`: Convierte vectores de Rodrigues a matrices de rotación.
- `rotation_matrix_to_rodrigues()`: Convierte matrices de rotación a vectores de Rodrigues.
- `project_point()`: Proyecta puntos 3D usando parámetros optimizados de la cámara.

### 5.5. Orquestador Principal (`perform_reconstruction.py`)

Clase principal que coordina todo el pipeline de reconstrucción 3D, integrando las etapas anteriores de forma eficiente.

**Funciones principales:**
- `start_3d_reconstruction()`: Función principal que ejecuta el pipeline completo con soporte para procesamiento paralelo.
- `calculate_session_extrinsics()`: Calcula extrínsecos robustos usando múltiples chunks.
- `reconstruct_frame_with_extrinsics()`: Reconstruye un frame usando extrínsecos pre-calculados.
- `save_3d_reconstruction()`: Guarda los resultados 3D en formato .npy.
- `process_chunk_parallel()`: Procesa chunks en paralelo para mayor velocidad.

**Características técnicas:**
- **Procesamiento paralelo**: Soporte para múltiples procesos para acelerar la reconstrucción.
- **Extrínsecos reutilizables**: Calcula una vez por sesión y reutiliza para todos los frames.
- **Escalado anatómico**: Normaliza automáticamente basándose en medidas antropométricas.
- **Gestión robusta de errores**: Manejo inteligente de frames con datos insuficientes.

### 5.6. Herramientas de Análisis

**`reprojection.py`**: Calcula errores de reproyección para evaluar la calidad de la reconstrucción.

**`analyze_3D_keypoint.py`**: Analiza keypoints 3D individuales y calcula medidas corporales detalladas.

**`complete_analysis.py`**: Herramienta de análisis completo que compara métodos de reconstrucción y muestra métricas biomecánicas.

---

Cada componente está diseñado para ser modular y eficiente, permitiendo la integración seamless con el pipeline principal del sistema de análisis de marcha.

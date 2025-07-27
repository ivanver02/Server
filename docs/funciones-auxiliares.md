# Funciones Auxiliares del Servidor

## Descripción General

Este documento detalla todas las funciones y clases que **NO están directamente involucradas en el pipeline principal** de procesamiento de video. Estas funciones proporcionan funcionalidades de soporte, utilidades, configuración y operaciones auxiliares.

## Análisis de Redundancia

**✅ NO se encontraron funciones redundantes** en el código del servidor. Todas las funciones tienen propósitos específicos y bien diferenciados.

---

## Funciones por Archivo

### `config/settings.py`

#### Clase `ServerConfig`
- **Propósito**: Configuración general del servidor Flask
- **Funciones auxiliares**:
  - Gestiona parámetros como host, puerto, debug, timeouts
  - Configuración de CORS y limites de archivos

#### Clase `DataConfig`
- **Propósito**: Configuración de directorios de datos
- **Función**: `ensure_directories()`
  - **Descripción**: Crea automáticamente los directorios necesarios para el almacenamiento de datos
  - **Ubicación**: `config/settings.py:32`

#### Clase `MMPoseConfig`
- **Propósito**: Configuración específica para modelos MMPose
- **Funciones auxiliares**:
  - `__post_init__()` - Inicialización automática post-creación
  - `ensure_directories()` - Garantiza existencia de directorios de modelos

#### Clase `ProcessingConfig`
- **Propósito**: Configuración de parámetros de procesamiento
- **Funciones auxiliares**:
  - Define umbrales de confianza, batch sizes, intervalos de frames

#### Clase `ReconstructionConfig`
- **Propósito**: Configuración para calibración y reconstrucción 3D
- **Funciones auxiliares**:
  - Parámetros de triangulación, métodos de calibración, tolerancias

---

### `config/camera_intrinsics.py`

#### `get_camera_intrinsics(camera_id: int)`
- **Propósito**: Obtiene parámetros intrínsecos de una cámara específica
- **Ubicación**: `config/camera_intrinsics.py:51`
- **Descripción**: Carga matriz de cámara y coeficientes de distorsión desde archivo

#### `get_default_intrinsics()`
- **Propósito**: Proporciona parámetros intrínsecos por defecto
- **Ubicación**: `config/camera_intrinsics.py:70`
- **Descripción**: Valores de respaldo cuando no hay calibración específica

#### `update_camera_intrinsics(camera_id: int, camera_matrix, distortion_coeffs)`
- **Propósito**: Actualiza y guarda nuevos parámetros intrínsecos
- **Ubicación**: `config/camera_intrinsics.py:87`
- **Descripción**: Persiste calibración intrínseca en archivo JSON

#### `get_all_camera_intrinsics()`
- **Propósito**: Obtiene todos los parámetros intrínsecos almacenados
- **Ubicación**: `config/camera_intrinsics.py:108`
- **Descripción**: Retorna diccionario completo de calibraciones

---

### `backend/processing/data/processing_result.py`

#### Clase `VideoProcessingResult`
- **Propósito**: Contenedor de datos para resultados de procesamiento de video individual
- **Función auxiliar**:
  - `__post_init__()` - Validación automática de resultados post-creación

#### Clase `MultiCameraResult`
- **Propósito**: Contenedor para resultados de múltiples cámaras sincronizadas
- **Función auxiliar**:
  - `__post_init__()` - Validación de consistencia entre cámaras

#### Clase `ProcessingSessionResult`
- **Propósito**: Contenedor para resultados completos de sesión
- **Función auxiliar**:
  - `__post_init__()` - Agregación y validación de resultados de sesión

---

### `backend/processing/detectors/base.py`

#### Clase `BasePoseDetector` (Abstracta)
- **Propósito**: Interfaz base para todos los detectores de pose
- **Funciones auxiliares**:
  - `get_model_info()` - Información del modelo cargado
  - `supports_video_processing()` - Capacidades del detector
  - `get_keypoint_names()` - Nombres de puntos clave del modelo
  - `get_num_keypoints()` - Número total de puntos clave

#### Clase `BaseDetectorManager` (Abstracta)
- **Propósito**: Gestor base para múltiples detectores
- **Funciones auxiliares**:
  - `register_detector()` - Registro de nuevos detectores
  - `get_detector()` - Obtención de detector específico
  - `get_available_detectors()` - Lista de detectores disponibles
  - `get_active_detectors()` - Lista de detectores activos
  - `cleanup_all()` - Limpieza de recursos de todos los detectores

#### Clase `DetectorFactory`
- **Propósito**: Factory pattern para creación de detectores
- **Función auxiliar**:
  - `create_detector()` - Método estático para instanciar detectores

---

### `backend/processing/detectors/mmpose/detector.py`

#### Clase `MMPoseDetector`

##### **Funciones de Inicialización y Configuración**
- `_is_cuda_available()` - Detección de disponibilidad CUDA
- `_find_model_files()` - Localización automática de archivos de modelo
- `get_model_info()` - Información detallada del modelo cargado
- `get_keypoint_names()` - Nombres específicos de keypoints del modelo
- `get_num_keypoints()` - Conteo de keypoints del modelo

##### **Funciones de Procesamiento Auxiliar**
- `detect_batch()` - Procesamiento por lotes (no usado en pipeline principal)
- `cleanup()` - Liberación de recursos del modelo

---

### `backend/processing/detectors/mmpose/manager.py`

#### Clase `MMPoseManager`
- **Propósito**: Gestor específico para modelos MMPose
- **Función auxiliar**:
  - `register_mmpose_model()` - Registro de modelos MMPose específicos
  - **Ubicación**: `backend/processing/detectors/mmpose/manager.py:25`

---

### `backend/processing/processors/multi_camera_processor.py`

#### Clase `MultiCameraProcessor`

##### **Funciones de Estado y Monitoreo**
- `get_status()` - Estado actual del procesador
- **Ubicación**: `backend/processing/processors/multi_camera_processor.py:245`
- **Descripción**: Información de rendimiento y estado del procesamiento

##### **Funciones de Procesamiento Interno**
- `_process_frame_parallel()` - Procesamiento paralelo de frames individuales
- **Ubicación**: `backend/processing/processors/multi_camera_processor.py:234`
- **Descripción**: Optimización para procesamiento multi-hilo

---

### `backend/processing/synchronization/video_sync.py`

#### Clase `VideoSynchronizer`

##### **Funciones de Información y Estado**
- `get_timestamps_list()` - Lista de timestamps disponibles
- **Ubicación**: `backend/processing/synchronization/video_sync.py:243`
- **Descripción**: Extrae todos los timestamps para análisis

- `get_sync_info()` - Información detallada de sincronización
- **Ubicación**: `backend/processing/synchronization/video_sync.py:272`
- **Descripción**: Estadísticas y metadatos de sincronización

##### **Funciones de Acceso Directo**
- `get_sync_frame_at_timestamp()` - Obtiene frame en timestamp específico
- **Ubicación**: `backend/processing/synchronization/video_sync.py:128`
- **Descripción**: Acceso aleatorio a frames sincronizados

#### Función Independiente
- `create_synchronizer_from_videos()` - Factory para crear sincronizador
- **Ubicación**: `backend/processing/synchronization/video_sync.py:321`
- **Descripción**: Función utilitaria para inicialización rápida

---

### `backend/processing/ensemble/ensemble_processor.py`

#### Clase `EnsembleResult`
- **Propósito**: Contenedor para resultados de ensemble
- **Descripción**: Estructura de datos para keypoints fusionados

#### Clase `EnsembleProcessor`

##### **Funciones de Configuración**
- `_load_config()` - Carga configuración de ensemble
- **Ubicación**: `backend/processing/ensemble/ensemble_processor.py:36`
- **Descripción**: Inicialización de parámetros de fusión

##### **Funciones de Carga de Datos**
- `load_keypoints_for_frame()` - Carga keypoints de frame específico
- **Ubicación**: `backend/processing/ensemble/ensemble_processor.py:60`
- **Descripción**: Acceso a datos previamente procesados

##### **Funciones de Fusión Específicas**
- `fuse_coco_keypoints()` - Fusión de keypoints COCO
- **Ubicación**: `backend/processing/ensemble/ensemble_processor.py:117`
- **Descripción**: Algoritmo específico para formato COCO

- `extract_additional_keypoints()` - Extracción de keypoints adicionales
- **Ubicación**: `backend/processing/ensemble/ensemble_processor.py:182`
- **Descripción**: Procesamiento de keypoints no-COCO

##### **Funciones de Persistencia**
- `save_ensemble_result()` - Guardado de resultados de ensemble
- **Ubicación**: `backend/processing/ensemble/ensemble_processor.py:322`
- **Descripción**: Serialización de resultados fusionados

##### **Funciones de Procesamiento Completo**
- `process_complete_session()` - Procesamiento de sesión completa
- **Ubicación**: `backend/processing/ensemble/ensemble_processor.py:361`
- **Descripción**: Ensemble para toda una sesión de grabación

---

### `backend/processing/coordinator.py`

#### Clase `ProcessingCoordinator`

##### **Funciones de Estado y Monitoreo**
- `get_status()` - Estado general del coordinador
- **Ubicación**: `backend/processing/coordinator.py:316`
- **Descripción**: Información de estado, progreso y recursos

##### **Funciones de Utilidad Interna**
- `_count_total_keypoints()` - Conteo de keypoints en resultado
- **Ubicación**: `backend/processing/coordinator.py:231`
- **Descripción**: Función auxiliar para estadísticas

- `_save_keypoints_2d()` - Guardado de keypoints 2D
- **Ubicación**: `backend/processing/coordinator.py:241`
- **Descripción**: Persistencia de resultados intermedios

---

### `backend/reconstruction/camera.py`

#### Clase `CameraCalibrationResult`
- **Propósito**: Contenedor para resultados de calibración
- **Descripción**: Estructura de datos para parámetros calibrados

#### Clase `Camera`

##### **Funciones de Inicialización**
- `_load_intrinsics()` - Carga parámetros intrínsecos
- **Ubicación**: `backend/reconstruction/camera.py:59`
- **Descripción**: Inicialización automática de parámetros

##### **Funciones de Configuración**
- `set_as_reference()` - Establece como cámara de referencia
- **Ubicación**: `backend/reconstruction/camera.py:73`
- **Descripción**: Configuración para sistema de coordenadas

- `set_extrinsics()` - Establece parámetros extrínsecos
- **Ubicación**: `backend/reconstruction/camera.py:87`
- **Descripción**: Rotación y translación respecto a referencia

##### **Funciones de Proyección y Validación**
- `project_3d_to_2d()` - Proyección de puntos 3D a 2D
- **Ubicación**: `backend/reconstruction/camera.py:138`
- **Descripción**: Para cálculo de error de reproyección

- `compute_reprojection_error()` - Cálculo de error de reproyección
- **Ubicación**: `backend/reconstruction/camera.py:171`
- **Descripción**: Validación de calidad de calibración

- `undistort_points()` - Corrección de distorsión
- **Ubicación**: `backend/reconstruction/camera.py:202`
- **Descripción**: Eliminación de distorsión radial/tangencial

##### **Funciones de Calibración**
- `calibrate_intrinsics_from_chessboard()` - Calibración con patrón de ajedrez
- **Ubicación**: `backend/reconstruction/camera.py:235`
- **Descripción**: Calibración automática de parámetros intrínsecos

##### **Funciones de Información**
- `get_summary()` - Resumen de parámetros de cámara
- **Ubicación**: `backend/reconstruction/camera.py:341`
- **Descripción**: Información completa de estado

##### **Funciones de Persistencia (No utilizadas)**
- `save_parameters()` - Guardado de parámetros
- **Ubicación**: `backend/reconstruction/camera.py:356`
- **Estado**: No usado actualmente

- `load_parameters()` - Carga de parámetros
- **Ubicación**: `backend/reconstruction/camera.py:375`
- **Estado**: No usado actualmente

---

### `backend/reconstruction/calibration.py`

#### Clase `CalibrationSystem`

##### **Funciones de Gestión de Cámaras**
- `add_camera()` - Registro de nueva cámara
- **Ubicación**: `backend/reconstruction/calibration.py:27`
- **Descripción**: Añade cámara al sistema de calibración

- `set_reference_camera()` - Establece cámara de referencia
- **Ubicación**: `backend/reconstruction/calibration.py:48`
- **Descripción**: Define origen del sistema de coordenadas

- `get_all_cameras()` - Obtiene todas las cámaras
- **Ubicación**: `backend/reconstruction/calibration.py:305`
- **Descripción**: Acceso al diccionario completo de cámaras

- `get_camera()` - Obtiene cámara específica
- **Ubicación**: `backend/reconstruction/calibration.py:309`
- **Descripción**: Acceso a cámara individual por ID

##### **Funciones de Calibración**
- `calibrate_camera_intrinsics()` - Calibración intrínseca
- **Ubicación**: `backend/reconstruction/calibration.py:68`
- **Descripción**: Delega en implementación de la cámara

- `calibrate_stereo_pair()` - Calibración estéreo entre dos cámaras
- **Ubicación**: `backend/reconstruction/calibration.py:108`
- **Descripción**: Calibración de par estéreo con OpenCV

##### **Funciones de Estado**
- `get_calibration_status()` - Estado de calibración del sistema
- **Ubicación**: `backend/reconstruction/calibration.py:313`
- **Descripción**: Información de estado de todas las cámaras

##### **Funciones de Persistencia**
- `save_calibration()` - Guardado de calibración completa
- **Ubicación**: `backend/reconstruction/calibration.py:330`
- **Descripción**: Serialización de todos los parámetros

- `load_calibration()` - Carga de calibración desde archivo
- **Ubicación**: `backend/reconstruction/calibration.py:354`
- **Descripción**: Restauración de calibración persistida

---

### `backend/reconstruction/triangulation.py`

#### Clase `TriangulationResult`
- **Propósito**: Contenedor para resultados de triangulación
- **Descripción**: Puntos 3D, errores, metadatos de calidad

#### Clase `Triangulator`

##### **Funciones de Configuración**
- `set_cameras()` - Configuración de cámaras (no usado)
- **Ubicación**: `backend/reconstruction/triangulation.py:33`
- **Estado**: No utilizado actualmente

##### **Funciones de Cálculo de Error**
- `_compute_point_reprojection_error()` - Error de reproyección de punto
- **Ubicación**: `backend/reconstruction/triangulation.py:52`
- **Descripción**: Validación de calidad de punto 3D individual

- `_compute_camera_reprojection_errors()` - Errores por cámara
- **Ubicación**: `backend/reconstruction/triangulation.py:88`
- **Descripción**: Análisis de calidad por cámara

##### **Funciones de Triangulación**
- `_triangulate_point_dlt()` - Triangulación DLT de punto individual
- **Ubicación**: `backend/reconstruction/triangulation.py:127`
- **Descripción**: Implementación Direct Linear Transform

##### **Funciones de Acceso a Datos**
- `_load_frame_keypoints()` - Carga keypoints de frame
- **Ubicación**: `backend/reconstruction/triangulation.py:387`
- **Descripción**: Acceso a datos de keypoints guardados

##### **Funciones de Persistencia**
- `save_triangulation_result()` - Guardado de resultado de triangulación
- **Ubicación**: `backend/reconstruction/triangulation.py:486`
- **Descripción**: Serialización de puntos 3D y metadatos

---

### `app.py`

#### **Funciones Auxiliares del Servidor Flask**

##### **Funciones de Triangulación Auxiliar**
- `_triangulate_chunk_simple()` - Triangulación simplificada de chunk
- **Ubicación**: `app.py:109`
- **Descripción**: Procesamiento 3D específico para chunks individuales

##### **Funciones de Validación**
- `_check_and_trigger_3d_reconstruction()` - Verificación para reconstrucción 3D
- **Ubicación**: `app.py:45`
- **Descripción**: Lógica de decisión para activar triangulación

##### **Manejadores de Error**
- `too_large(e)` - Manejo de archivos demasiado grandes
- **Ubicación**: `app.py:503`
- **Descripción**: Error handler para límites de tamaño

- `not_found(e)` - Manejo de recursos no encontrados
- **Ubicación**: `app.py:508`
- **Descripción**: Error handler 404

- `internal_error(e)` - Manejo de errores internos
- **Ubicación**: `app.py:513`
- **Descripción**: Error handler 500

---

## Resumen de Funcionalidades Auxiliares

### **Por Categoría Funcional**

1. **Configuración y Setup**
   - Gestión de directorios y archivos
   - Parámetros de modelos y procesamiento
   - Configuración de calibración

2. **Validación y Calidad**
   - Cálculo de errores de reproyección
   - Validación de resultados
   - Métricas de calidad

3. **Persistencia y Almacenamiento**
   - Guardado y carga de calibraciones
   - Serialización de resultados
   - Gestión de archivos de datos

4. **Información y Monitoreo**
   - Estados de sistemas y procesos
   - Estadísticas y metadatos
   - Información de modelos

5. **Utilidades de Procesamiento**
   - Funciones helper para algoritmos
   - Acceso a datos específicos
   - Operaciones de transformación

### **Estado de Redundancia**

✅ **Confirmado**: No existen funciones redundantes en el código del servidor. Cada función tiene un propósito específico y bien definido dentro de su contexto modular.

---

*Documento generado automáticamente. Última actualización: $(Get-Date)*

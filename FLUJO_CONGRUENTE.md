"""
RESUMEN DE FLUJO Y ESTRUCTURA CONGRUENTE DEL SISTEMA
=====================================================

## FLUJO COMPLETO DE UN CHUNK:

1. **RECEPCIÓN** (/api/chunks/receive)
   - Cliente sube archivo MP4 por cámara
   - Se guarda en: unprocessed_dir/patient{id}/session{id}/camera{id}/{chunk}.mp4
   - Verifica si todos los chunks están disponibles

2. **TRIGGER DE PROCESAMIENTO**
   - Cuando todas las cámaras tienen el chunk → procesamiento inmediato
   - Usa video_pipeline.process_chunk_synchronized()

3. **SINCRONIZACIÓN**
   - VideoSynchronizer alinea videos por timestamp y FPS
   - Extrae frames sincronizados de todas las cámaras

4. **DETECCIÓN DE POSE** (MultiCameraProcessor)
   - Por cada frame sincronizado:
     - VitPose.detect_frame() → (keypoints: np.ndarray(N,2), scores: np.ndarray(N,))
     - HRNet-W48.detect_frame() → (keypoints: np.ndarray(N,2), scores: np.ndarray(N,))
     - WholeBody.detect_frame() → (keypoints: np.ndarray(N,2), scores: np.ndarray(N,))
     - RTMPose.detect_frame() → (keypoints: np.ndarray(N,2), scores: np.ndarray(N,))

5. **GUARDADO ESTRUCTURADO** (save_keypoints_2d_frame)
   ```
   data/patient{id}/session{id}/camera{id}/
   ├── keypoints/
   │   ├── {global_frame}_VitPose.npy          # np.ndarray(N,2)
   │   ├── {global_frame}_HRNet-W48.npy        # np.ndarray(N,2)
   │   ├── {global_frame}_WholeBody.npy        # np.ndarray(N,2)
   │   └── {global_frame}_RTMPose.npy          # np.ndarray(N,2)
   └── confidence/
       ├── {global_frame}_VitPose.npy          # np.ndarray(N,)
       ├── {global_frame}_HRNet-W48.npy        # np.ndarray(N,)
       ├── {global_frame}_WholeBody.npy        # np.ndarray(N,)
       └── {global_frame}_RTMPose.npy          # np.ndarray(N,)
   ```

6. **TRIANGULACIÓN 3D** (Triangulator)
   - Se activa automáticamente cuando hay suficientes datos 2D
   - Usa load_keypoints_2d_frame() para cargar datos
   - Genera puntos 3D con información de confianza

## ESTRUCTURAS DE DATOS CONGRUENTES:

### DETECTORES (MMPose)
```python
class BasePoseDetector:
    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Retorna: (keypoints(N,2), scores(N,))
```

### PROCESAMIENTO
```python
class FrameResult:
    frame_number: int
    timestamp: float
    camera_id: int
    frame_processed: bool           # Si se procesó exitosamente
    detectors_used: List[str]       # ['VitPose', 'HRNet-W48', ...]
    num_detections: int             # Número de detectores exitosos
```

### FUNCIONES AUXILIARES
```python
# Guardar datos separados
save_keypoints_2d_frame(keypoints, confidence, detector_name, camera_id, global_frame, patient_id, session_id)

# Cargar datos separados
load_keypoints_2d_frame(detector_name, camera_id, global_frame, patient_id, session_id) 
# → (keypoints: np.ndarray(N,2), confidence: np.ndarray(N,))

# Análisis de sesión
get_session_summary(patient_id, session_id)
get_available_frames(camera_id, patient_id, session_id)
get_available_detectors(camera_id, patient_id, session_id)
```

## ENDPOINTS API:

- POST `/api/chunks/receive` - Recibir chunk de video
- GET  `/api/pipeline/status` - Estado del pipeline
- GET  `/api/session/{patient_id}/{session_id}/analysis` - Análisis de sesión

## PRINCIPIOS DE CONGRUENCIA IMPLEMENTADOS:

1. **SEPARACIÓN CLARA**: Keypoints y confianza en archivos separados
2. **FORMATO DIRECTO**: np.ndarray tal como lo devuelve MMPose
3. **SIN ABSTRACCIONES INNECESARIAS**: No KeypointResult para 2D
4. **FUNCIONES AUXILIARES ESPECÍFICAS**: Una responsabilidad por función
5. **ESTRUCTURA ORGANIZADA**: Carpetas separadas por tipo de dato
6. **FLUJO LINEAL**: Chunk → Sync → Detect → Save → Triangulate

## ARCHIVOS PRINCIPALES:

- `pipeline.py` - Flujo principal documentado
- `utils/keypoint_io.py` - Funciones auxiliares de I/O
- `processors/multi_camera_processor.py` - Procesamiento multi-cámara
- `data/keypoint_result.py` - Estructuras de datos simplificadas
- `app.py` - API endpoints actualizados
- `reconstruction/triangulation.py` - Triangulación usando nuevas funciones

## TESTING:

Para verificar la congruencia:
```python
from backend.processing.utils import save_keypoints_2d_frame, load_keypoints_2d_frame

# Simular detección
keypoints = np.random.rand(17, 2)  # 17 keypoints COCO
scores = np.random.rand(17)        # Confianza por keypoint

# Guardar
save_keypoints_2d_frame(keypoints, scores, 'VitPose', 0, 1001, 'p1', 's1')

# Cargar
loaded_kp, loaded_conf = load_keypoints_2d_frame('VitPose', 0, 1001, 'p1', 's1')

# Verificar
assert np.array_equal(keypoints, loaded_kp)
assert np.array_equal(scores, loaded_conf)
```

RESULTADO: Sistema completamente congruente donde solo se guardan coordenadas y confianzas por separado, tal como las devuelve MMPose, sin abstracciones innecesarias.
"""

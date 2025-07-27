# Pipeline de Ejecución - Proyecto Server

## 📋 Descripción General

Este documento describe el flujo completo de procesamiento en el servidor desde que recibe un chunk de video hasta que genera keypoints 3D finales. El servidor procesa video usando MMPose, aplica ensemble learning y realiza triangulación estéreo para reconstrucción 3D.

## 🔄 Flujo Principal de Ejecución

### **FASE 1: RECEPCIÓN DE CHUNKS**

#### 1.1. `POST /api/chunks/receive`
**Archivo:** `app.py:335`
**Descripción:** Endpoint principal que recibe chunks de video desde el cliente Code con metadatos.

#### 1.2. Validación de sesión activa
**Archivo:** `app.py:348`
**Descripción:** Verifica que existe una sesión de procesamiento activa antes de aceptar chunks.

#### 1.3. Validación de archivo y metadatos
**Archivo:** `app.py:355-375`
**Descripción:** Valida que el archivo MP4 existe y que camera_id y chunk_number son válidos.

#### 1.4. Guardado de archivo
**Archivo:** `app.py:380-390`
**Descripción:** Guarda el chunk MP4 en directorio estructurado por paciente/sesión/cámara.

#### 1.5. Verificación de chunks sincronizados
**Archivo:** `app.py:395-405`
**Descripción:** Verifica si están disponibles todos los chunks del mismo número de todas las cámaras.

---

### **FASE 2: PROCESAMIENTO INMEDIATO (Si todos los chunks están disponibles)**

#### 2.1. Inicialización del coordinador
**Archivo:** `app.py:410-420`
**Descripción:** Inicializa el processing_coordinator que gestiona MMPose y ensemble learning.

##### 2.1.1. `processing_coordinator.initialize()`
**Archivo:** `backend/processing/coordinator.py:39`
**Descripción:** Inicializa detector manager y multi-camera processor.

##### 2.1.2. `detector_manager.initialize_all()`
**Archivo:** `backend/processing/coordinator.py:60`
**Descripción:** Carga todos los detectores MMPose configurados.

##### 2.1.3. `multi_camera_processor.initialize()`
**Archivo:** `backend/processing/coordinator.py:67`
**Descripción:** Inicializa procesador multi-cámara con detector manager.

#### 2.2. `processing_coordinator.process_chunk_videos()`
**Archivo:** `backend/processing/coordinator.py:85` → `app.py:420`
**Descripción:** Procesa todos los videos del chunk desde todas las cámaras simultáneamente.

##### 2.2.1. `multi_camera_processor.process_synchronized_videos()`
**Archivo:** `backend/processing/processors/multi_camera_processor.py:57`
**Descripción:** Procesa múltiples videos sincronizados usando VideoSynchronizer.

###### 2.2.1.1. `create_synchronizer_from_videos()`
**Archivo:** `backend/processing/synchronization/video_sync.py:321`
**Descripción:** Crea sincronizador de videos desde múltiples archivos de video.

###### 2.2.1.2. `synchronizer.initialize_sync()`
**Archivo:** `backend/processing/synchronization/video_sync.py:88`
**Descripción:** Inicializa sincronización calculando timestamps y metadatos.

###### 2.2.1.3. `_process_synchronized_frames()`
**Archivo:** `backend/processing/processors/multi_camera_processor.py:166`
**Descripción:** Procesa frames sincronizados usando iterator de sincronización.

####### 2.2.1.3.1. `synchronizer.iterate_synchronized_frames()`
**Archivo:** `backend/processing/synchronization/video_sync.py:200`
**Descripción:** Itera sobre frames sincronizados de todas las cámaras.

####### 2.2.1.3.2. `_process_frame_parallel()` (para cada frame)
**Archivo:** `backend/processing/processors/multi_camera_processor.py:234`
**Descripción:** Procesa frame individual con MMPose usando ThreadPoolExecutor.

######## 2.2.1.3.2.1. `detector.detect_frame()` (para cada detector)
**Archivo:** `backend/processing/detectors/mmpose/detector.py:106`
**Descripción:** Ejecuta inferencia MMPose en frame individual con detector específico.

######## 2.2.1.3.2.2. Guardado de keypoints 2D
**Archivo:** Interno en `_process_frame_parallel`
**Descripción:** Guarda keypoints 2D y confidence scores por detector en archivos .npy.

#### 2.3. `_save_keypoints_2d()`
**Archivo:** `backend/processing/coordinator.py:241`
**Descripción:** Consolida y guarda resultados de keypoints 2D de todas las cámaras.

#### 2.4. `_apply_ensemble_to_chunk()` (Si hay múltiples detectores)
**Archivo:** `backend/processing/coordinator.py:184`
**Descripción:** Aplica ensemble learning combinando resultados de múltiples detectores.

##### 2.4.1. `ensemble_processor.process_frame_ensemble()` (para cada frame)
**Archivo:** `backend/processing/ensemble/ensemble_processor.py:244`
**Descripción:** Combina keypoints de múltiples detectores usando weighted average.

##### 2.4.2. `ensemble_processor.save_ensemble_result()`
**Archivo:** `backend/processing/ensemble/ensemble_processor.py:322`
**Descripción:** Guarda keypoints finales del ensemble en formato numpy.

---

### **FASE 3: VERIFICACIÓN PARA RECONSTRUCCIÓN 3D**

#### 3.1. `_check_and_trigger_3d_reconstruction()`
**Archivo:** `app.py:430` → `app.py:45`
**Descripción:** Verifica si están disponibles keypoints 2D de todas las cámaras para triangulación.

#### 3.2. Verificación de keypoints 2D disponibles
**Archivo:** `app.py:55-78`
**Descripción:** Busca archivos .npy de keypoints en directorios de todas las cámaras.

#### 3.3. `_triangulate_chunk_simple()` (Si todos los keypoints están disponibles)
**Archivo:** `app.py:109`
**Descripción:** Ejecuta triangulación simple para generar keypoints 3D (placeholder).

##### 3.3.1. Carga de keypoints por cámara
**Archivo:** `app.py:115-140`
**Descripción:** Carga archivos .npy de keypoints 2D de todas las cámaras.

##### 3.3.2. Guardado de resultados 3D
**Archivo:** `app.py:150-170`
**Descripción:** Guarda resultados 3D en formato JSON (placeholder para triangulación real).

---

### **FASE 4: ENDPOINTS DE GESTIÓN DE SESIÓN**

#### 4.1. `POST /api/session/start`
**Archivo:** `app.py:208`
**Descripción:** Inicia nueva sesión de procesamiento y crea estructura de directorios.

##### 4.1.1. Validación de parámetros
**Archivo:** `app.py:220-235`
**Descripción:** Valida patient_id, session_id y cameras_count requeridos.

##### 4.1.2. Creación de directorios
**Archivo:** `app.py:245-260`
**Descripción:** Crea estructura de directorios para cada cámara de la sesión.

##### 4.1.3. Activación de sesión global
**Archivo:** `app.py:262-268`
**Descripción:** Actualiza variable global current_session con información de la nueva sesión.

#### 4.2. `POST /api/session/cancel`
**Archivo:** `app.py:280`
**Descripción:** Cancela sesión activa y limpia todos los archivos generados.

##### 4.2.1. Limpieza de directorios
**Archivo:** `app.py:295-320`
**Descripción:** Elimina directorios de datos de la sesión cancelada.

##### 4.2.2. Reset de sesión global
**Archivo:** `app.py:322-330`
**Descripción:** Reinicia variable global current_session a estado inactivo.

#### 4.3. `POST /api/cameras/recalibrate`
**Archivo:** `app.py:444`
**Descripción:** Recalibra parámetros extrínsecos de cámaras usando keypoints de la sesión.

##### 4.3.1. `calibration_system.auto_calibrate_extrinsics_from_session()`
**Archivo:** `backend/reconstruction/calibration.py:192`
**Descripción:** Auto-calibración usando keypoints 2D correspondientes entre cámaras.

##### 4.3.2. `calibration_system.save_calibration()`
**Archivo:** `backend/reconstruction/calibration.py:330`
**Descripción:** Guarda parámetros de calibración actualizados en archivo .npz.

---

### **FASE 5: ENDPOINTS DE ESTADO Y SALUD**

#### 5.1. `GET /health`
**Archivo:** `app.py:199`
**Descripción:** Endpoint de verificación de salud del servidor.

#### 5.2. `GET /api/session/status`
**Archivo:** `app.py:272`
**Descripción:** Retorna estado actual de la sesión de procesamiento.

---

## 📊 Flujo de Datos

### **Estructura de Archivos Generados:**

```
data/unprocessed/
├── patient{ID}/
    └── session{ID}/
        ├── camera0/
        │   ├── 0.mp4, 1.mp4, 2.mp4...    # Chunks originales
        │   ├── 2D/
        │   │   ├── points/{modelo}/      # Keypoints por modelo
        │   │   │   └── {frame}_{chunk}.npy
        │   │   └── confidence/{modelo}/  # Confidence por modelo
        │   │       └── {frame}_{chunk}_confidence.npy
        │   └── ensemble/                 # Resultados de ensemble
        │       ├── {frame}_{chunk}.npy
        │       └── {frame}_{chunk}_confidence.npy
        ├── camera1/ (misma estructura)
        └── camera2/ (misma estructura)

data/processed/keypoints_3d/
└── patient{ID}/
    └── session{ID}/
        └── {frame}_{chunk}_3d.npy       # Keypoints 3D triangulados
```

### **Flujo de Procesamiento por Frame:**

```
Video MP4 → Extracción de Frames → MMPose (múltiples modelos)
    ↓
Keypoints 2D + Confidence (por modelo) → Ensemble Learning
    ↓
Keypoints 2D Finales → Triangulación Estéreo
    ↓
Keypoints 3D Finales
```

## ⚡ Características Clave

- **Procesamiento Inmediato:** Se procesa en cuanto están disponibles todos los chunks sincronizados
- **Multi-Modelo MMPose:** Utiliza múltiples modelos COCO y Extended simultáneamente
- **Ensemble Learning:** Combina resultados de múltiples modelos para mayor precisión
- **Triangulación Automática:** Genera keypoints 3D automáticamente cuando hay datos 2D completos
- **Gestión de Sesiones:** Control completo del ciclo de vida de sesiones de procesamiento
- **Auto-Calibración:** Recalibración automática de cámaras usando keypoints de la sesión
- **Procesamiento Sincronizado:** Espera chunks de todas las cámaras antes de procesar
- **Limpieza Automática:** Eliminación de archivos temporales y sesiones canceladas

## 🔄 Resumen del Flujo Completo

```
Code envía chunk → /api/chunks/receive
         ↓
¿Todos los chunks del mismo número disponibles?
         ↓ (Sí)
processing_coordinator.process_chunk_videos()
         ↓
Para cada cámara: Video → Frames → MMPose → Ensemble
         ↓
Guardar keypoints 2D de cada cámara
         ↓
¿Keypoints 2D de todas las cámaras disponibles?
         ↓ (Sí)
Triangulación estéreo → Keypoints 3D
         ↓
Guardar keypoints 3D finales
```

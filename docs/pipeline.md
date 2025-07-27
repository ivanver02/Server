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
**Archivo:** `app.py:351-371`
**Descripción:** Valida que el archivo MP4 existe y que camera_id y chunk_number son válidos.

#### 1.4. Guardado de archivo
**Archivo:** `app.py:382-390`
**Descripción:** Guarda el chunk MP4 en directorio estructurado por paciente/sesión/cámara.

#### 1.5. Verificación de chunks sincronizados
**Archivo:** `app.py:395-404`
**Descripción:** Verifica si están disponibles todos los chunks del mismo número de todas las cámaras.

---

### **FASE 2: PROCESAMIENTO INMEDIATO (Si todos los chunks están disponibles)**

#### 2.1. Inicialización del coordinador
**Archivo:** `app.py:408-413`
**Descripción:** Inicializa el processing_coordinator que gestiona MMPose y ensemble learning.

##### 2.1.1. `processing_coordinator.initialize()`
**Archivo:** `backend/processing/coordinator.py:52`
**Descripción:** Inicializa MMPose wrapper y ensemble processor para procesamiento de video.

##### 2.1.2. `mmpose_wrapper.initialize()`
**Archivo:** `backend/processing/coordinator.py:66`
**Descripción:** Carga todos los modelos MMPose configurados (COCO y Extended).

##### 2.1.3. `ensemble_processor.initialize()`
**Archivo:** `backend/processing/coordinator.py:73`
**Descripción:** Inicializa sistema de ensemble learning para combinar resultados de múltiples modelos.

#### 2.2. `processing_coordinator.process_chunk_videos()`
**Archivo:** `backend/processing/coordinator.py:82` → `app.py:415`
**Descripción:** Procesa todos los videos del chunk desde todas las cámaras simultáneamente.

##### 2.2.1. Creación de objetos Video
**Archivo:** `backend/processing/coordinator.py:106`
**Descripción:** Crea objetos Video individuales para cada archivo de cámara con metadatos completos.

##### 2.2.2. `video.process_complete_pipeline()` (para cada cámara)
**Archivo:** `backend/data/video.py:89`
**Descripción:** Pipeline completo: extracción de frames → MMPose → ensemble → guardado de keypoints 2D.

###### 2.2.2.1. `video.extract_all_frames()`
**Archivo:** `backend/data/video.py:106`
**Descripción:** Extrae todos los frames del video MP4 como imágenes temporales.

###### 2.2.2.2. `video.process_all_models()`
**Archivo:** `backend/data/video.py:127`
**Descripción:** Procesa cada frame con todos los modelos MMPose configurados.

####### 2.2.2.2.1. `inferencer.forward()` (para cada modelo)
**Archivo:** `backend/data/video.py:134`
**Descripción:** Ejecuta inferencia MMPose en cada frame con cada modelo individual.

####### 2.2.2.2.2. `video._save_model_results()`
**Archivo:** `backend/data/video.py:149`
**Descripción:** Guarda keypoints 2D y confidence scores de cada modelo por separado.

###### 2.2.2.3. `video.apply_ensemble_to_all_frames()`
**Archivo:** `backend/data/video.py:170`
**Descripción:** Aplica ensemble learning a todos los frames procesados.

####### 2.2.2.3.1. `ensemble_processor.process_frame_ensemble()`
**Archivo:** `backend/processing/ensemble_processor.py:96`
**Descripción:** Combina resultados de múltiples modelos usando weighted average y confidence filtering.

####### 2.2.2.3.2. `ensemble_processor.save_ensemble_result()`
**Archivo:** `backend/processing/ensemble_processor.py:145`
**Descripción:** Guarda keypoints 2D finales del ensemble en formato numpy.

###### 2.2.2.4. `video.cleanup_temp_images()`
**Archivo:** `backend/data/video.py:191`
**Descripción:** Limpia imágenes temporales extraídas del video.

#### 2.3. Resultado de procesamiento por chunk
**Archivo:** `backend/processing/coordinator.py:130`
**Descripción:** Consolida resultados de todas las cámaras y retorna estadísticas completas.

---

### **FASE 3: VERIFICACIÓN PARA RECONSTRUCCIÓN 3D**

#### 3.1. `_check_and_trigger_3d_reconstruction()`
**Archivo:** `app.py:425` → `app.py:43`
**Descripción:** Verifica si están disponibles keypoints 2D de todas las cámaras para triangulación.

#### 3.2. Verificación de keypoints 2D disponibles
**Archivo:** `app.py:55-78`
**Descripción:** Busca archivos .npy de keypoints en directorios de todas las cámaras.

#### 3.3. `_triangulate_chunk_simple()` (Si todos los keypoints están disponibles)
**Archivo:** `app.py:81`
**Descripción:** Ejecuta triangulación estéreo para generar keypoints 3D.

##### 3.3.1. `Triangulator.triangulate_multiple_frames()`
**Archivo:** `backend/reconstruction/triangulation.py:89`
**Descripción:** Triangula keypoints de múltiples frames usando geometría estéreo.

##### 3.3.2. Guardado de keypoints 3D
**Archivo:** `app.py:143-162`
**Descripción:** Guarda keypoints 3D finales en formato numpy en directorio estructurado.

---

### **FASE 4: ENDPOINTS DE GESTIÓN DE SESIÓN**

#### 4.1. `POST /api/session/start`
**Archivo:** `app.py:206`
**Descripción:** Inicia nueva sesión de procesamiento y crea estructura de directorios.

##### 4.1.1. Validación de parámetros
**Archivo:** `app.py:222-234`
**Descripción:** Valida patient_id, session_id y cameras_count requeridos.

##### 4.1.2. Creación de directorios
**Archivo:** `app.py:240-256`
**Descripción:** Crea estructura completa de directorios para la sesión (2D/points, 2D/confidence por modelo).

##### 4.1.3. Activación de sesión global
**Archivo:** `app.py:259-264`
**Descripción:** Actualiza variable global current_session con información de la nueva sesión.

#### 4.2. `POST /api/session/cancel`
**Archivo:** `app.py:289`
**Descripción:** Cancela sesión activa y limpia todos los archivos generados.

##### 4.2.1. Limpieza de directorios
**Archivo:** `app.py:305-320`
**Descripción:** Elimina directorios de datos sin procesar y procesados de la sesión cancelada.

##### 4.2.2. Reset de sesión global
**Archivo:** `app.py:324-330`
**Descripción:** Reinicia variable global current_session a estado inactivo.

#### 4.3. `POST /api/cameras/recalibrate`
**Archivo:** `app.py:450`
**Descripción:** Recalibra parámetros extrínsecos de cámaras usando keypoints de la sesión.

##### 4.3.1. `calibration_system.auto_calibrate_extrinsics_from_session()`
**Archivo:** `backend/reconstruction/calibration.py:245`
**Descripción:** Auto-calibración usando keypoints 2D correspondientes entre cámaras.

##### 4.3.2. `calibration_system.save_calibration()`
**Archivo:** `backend/reconstruction/calibration.py:198`
**Descripción:** Guarda parámetros de calibración actualizados en archivo .npz.

---

### **FASE 5: ENDPOINTS DE ESTADO Y SALUD**

#### 5.1. `GET /health`
**Archivo:** `app.py:197`
**Descripción:** Endpoint de verificación de salud del servidor.

#### 5.2. `GET /api/session/status`
**Archivo:** `app.py:279`
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

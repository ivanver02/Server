# 🦴 GUÍA COMPLETA DEL SISTEMA DE ANÁLISIS DE MARCHA

**Guía paso a paso para entender y usar todo el código generado**

## 📋 Resumen Ejecutivo del Sistema

Este es un **sistema completo de análisis de marcha** para detectar gonartrosis mediante:

1. **Recepción de videos** del cliente (repositorio Code)
2. **Procesamiento con 4 modelos MMPose** para máxima precisión
3. **Reconstrucción 3D** de keypoints mediante triangulación
4. **Análisis cinemático** de patrones de movimiento

**Total: ~5,800 líneas de código Python + documentación**

## 🗺️ ESTRUCTURA DEL PROYECTO

```
Server/                                 # Repositorio principal
├── 🚀 ARCHIVOS PRINCIPALES
│   ├── app.py                         # ⭐ SERVIDOR FLASK PRINCIPAL (292 líneas)
│   ├── quick_start.py                 # ⭐ SCRIPT DE INICIO RÁPIDO (98 líneas)
│   ├── download_models.py             # ⭐ DESCARGADOR DE MODELOS (305 líneas)
│   ├── verify_system.py               # ⭐ VERIFICADOR DEL SISTEMA (421 líneas)
│   └── requirements.txt               # Dependencias Python
│
├── 🎛️ CONFIGURACIÓN
│   ├── config/
│   │   ├── settings.py                # Configuración principal (152 líneas)
│   │   ├── camera_intrinsics.py       # Parámetros de cámaras (121 líneas)
│   │   └── keypoint_mappings.py       # Mapeo de keypoints (138 líneas)
│
├── 🧠 PROCESAMIENTO CORE
│   ├── backend/processing/
│   │   ├── pose_detector.py           # Wrapper MMPose (261 líneas)
│   │   ├── ensemble_processor.py      # Fusión de modelos (439 líneas)
│   │   ├── video_processor.py         # Procesamiento video (356 líneas)
│   │   └── models/                    # Modelos de datos
│   │       ├── photo.py               # Modelo de fotos (230 líneas)
│   │       └── video.py               # Modelo de videos (386 líneas)
│
├── 🔺 RECONSTRUCCIÓN 3D
│   ├── backend/reconstruction/
│   │   ├── triangulation.py           # ⭐ TRIANGULACIÓN DLT (525 líneas)
│   │   ├── calibration.py             # Sistema calibración (471 líneas)
│   │   └── camera.py                  # Modelo de cámara (401 líneas)
│
├── 🧪 SISTEMA DE TESTING
│   ├── backend/tests/
│   │   ├── run_all_tests.py           # ⭐ TEST RUNNER PRINCIPAL (301 líneas)
│   │   ├── test_mmpose.py             # Tests MMPose (353 líneas)
│   │   ├── test_calibration.py        # Tests calibración (440 líneas)
│   │   ├── test_triangulation.py      # Tests triangulación (461 líneas)
│   │   ├── test_reconstruction.py     # Tests completos (533 líneas)
│   │   └── README.md                  # Documentación tests
│
├── 🗃️ DATOS (gitignored)
│   ├── data/unprocessed/              # Videos del cliente
│   ├── data/processed/                # Datos procesados
│   └── mmpose_models/                 # Modelos MMPose descargados
│
└── 📚 DOCUMENTACIÓN
    ├── README.md                      # ⭐ DOCUMENTACIÓN PRINCIPAL
    └── .gitignore                     # Archivos ignorados
```

## 🚀 DÓNDE EMPEZAR: ORDEN DE LECTURA

### 1️⃣ **PRIMERA LECTURA - Entender el Sistema**

```bash
# Lee estos archivos en este orden:
📖 README.md                          # Visión general del proyecto
📖 config/settings.py                 # Configuración y parámetros
📖 app.py                             # API principal y endpoints
📖 backend/tests/README.md            # Cómo funciona el testing
```

### 2️⃣ **SEGUNDA LECTURA - Pipeline de Procesamiento**

```bash
# Flujo de datos paso a paso:
📖 backend/processing/video_processor.py       # 1. Procesamiento de video
📖 backend/processing/pose_detector.py         # 2. Detección de poses MMPose
📖 backend/processing/ensemble_processor.py    # 3. Fusión de 4 modelos
📖 backend/reconstruction/triangulation.py     # 4. Reconstrucción 3D
```

### 3️⃣ **TERCERA LECTURA - Sistemas de Soporte**

```bash
# Sistemas auxiliares:
📖 backend/reconstruction/calibration.py       # Calibración de cámaras
📖 backend/reconstruction/camera.py            # Modelo de cámara
📖 config/camera_intrinsics.py                # Parámetros de cámaras
📖 config/keypoint_mappings.py                # Mapeo de keypoints
```

### 4️⃣ **CUARTA LECTURA - Testing y Validación**

```bash
# Sistema de testing:
📖 backend/tests/run_all_tests.py             # Test runner principal
📖 backend/tests/test_triangulation.py        # Tests más importantes
📖 backend/tests/test_mmpose.py               # Tests de modelos
```

## 🎯 FUNCIONAMIENTO GENERAL

### 📊 **FLUJO DE DATOS PRINCIPAL**

```
1. RECEPCIÓN (app.py)
   Chunks de video del cliente → POST /api/chunks/receive

2. EXTRACCIÓN DE FRAMES (video_processor.py)
   Video MP4 → Frames individuales (15 FPS)

3. DETECCIÓN 2D (pose_detector.py)
   Frames → 4 modelos MMPose → Keypoints 2D + confianza

4. ENSEMBLE LEARNING (ensemble_processor.py)
   4 resultados → Fusión ponderada → Keypoints 2D finales

5. TRIANGULACIÓN 3D (triangulation.py)
   Keypoints 2D multi-cámara → DLT → Puntos 3D

6. VALIDACIÓN (calibration.py)
   Puntos 3D → Error reproyección → Filtrado

7. ALMACENAMIENTO
   Keypoints 3D → data/processed/
```

### 🏗️ **ARQUITECTURA DE MODELOS**

```python
# 4 Modelos MMPose con distribución GPU:

GPU:0 (cuda:0) - Modelos COCO principales:
├── HRNet-W48 (peso: 0.6) - Máxima precisión
└── HRNet-W32 (peso: 0.4) - Complementario

GPU:1 (cuda:1) - Modelos extendidos:
├── ResNet50-RLE (peso: 1.0) - Robustez
└── WholeBody (peso: 1.0) - Keypoints pies
```

### 🔄 **CICLO DE VIDA DE UNA SESIÓN**

```python
# 1. Cliente inicia sesión
POST /api/session/start
├── patient_id: "patient_001"
├── session_id: UUID generado
└── camera_count: 3-5 cámaras

# 2. Cliente envía chunks
POST /api/chunks/receive
├── Chunk video 5 segundos
├── Procesamiento automático
└── Keypoints 3D generados

# 3. Cliente cancela (opcional)
POST /api/session/cancel
├── Limpieza archivos temp
└── Preservar datos procesados
```

## 📁 DÓNDE SE GUARDA CADA COSA

### 🗂️ **ESTRUCTURA DE DATOS**

```bash
data/
├── unprocessed/                    # 📥 ENTRADA (del cliente)
│   └── patient001/session123/
│       ├── camera0/
│       │   ├── 1.mp4, 2.mp4...    # Chunks de video
│       │   └── 2D/                # Keypoints por modelo
│       │       ├── points/hrnet_w48/1.npy
│       │       └── confidence/hrnet_w48/1.npy
│       ├── camera1/...
│       └── camera2/...
│
└── processed/                      # 📤 SALIDA (procesado)
    ├── photos_from_video/         # 🖼️ Frames extraídos (temporal)
    ├── 2D_keypoints/              # 🎯 Keypoints 2D finales
    │   └── patient001/session123/camera0/
    │       ├── 1.npy              # Coordenadas (x,y) finales
    │       └── 1_confidence.npy   # Confianzas finales
    └── 3D_keypoints/              # 🔺 RECONSTRUCCIÓN 3D
        └── patient001/session123/
            ├── 1.npy              # Puntos 3D (x,y,z)
            ├── 1_confidence.npy   # Confianzas 3D
            └── 1_info.npz         # Metadatos (error, etc.)
```

### 🎛️ **CONFIGURACIÓN**

```python
# config/settings.py - CONFIGURACIÓN PRINCIPAL
class ProcessingConfig:
    primary_gpu = 'cuda:0'          # GPU para HRNet
    secondary_gpu = 'cuda:1'        # GPU para ResNet/WholeBody
    target_fps = 15                 # FPS procesamiento
    confidence_threshold = 0.3      # Filtro confianza

class APIConfig:
    host = '0.0.0.0'               # Host servidor
    port = 5000                    # Puerto servidor
    max_chunk_size = 100 * 1024 * 1024  # 100MB chunks

# config/camera_intrinsics.py - PARÁMETROS CÁMARAS
CAMERA_INTRINSICS = {
    0: {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0},
    1: {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0},
    # ...
}

# config/keypoint_mappings.py - MAPEO KEYPOINTS
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    # ... 17 puntos COCO
]
```

### 🧠 **MODELOS MMPOSE**

```bash
mmpose_models/
├── configs/pose2d/                # 📄 Archivos configuración (.py)
│   ├── td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py
│   ├── td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192.py
│   ├── td-hm_res50_rle-8xb64-210e_coco-256x192.py
│   └── td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288.py
│
└── checkpoints/                   # 🧠 Pesos del modelo (.pth)
    ├── td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-1d86a0de_20220909.pth
    ├── td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192-81c58e40_20220909.pth
    ├── td-hm_res50_rle-8xb64-210e_coco-256x192-c4aa2b08_20220913.pth
    └── td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288-ce11e65b_20220913.pth
```

## 🚀 CÓMO EMPEZAR A USAR EL SISTEMA

### 🔧 **INSTALACIÓN Y CONFIGURACIÓN**

```bash
# 1. Verificar sistema
python verify_system.py

# 2. Instalar dependencias (si faltan)
pip install -r requirements.txt

# 3. Descargar modelos MMPose
python download_models.py

# 4. Ejecutar tests (opcional pero recomendado)
python -m backend.tests.run_all_tests --quick

# 5. Iniciar servidor
python quick_start.py
# o directamente:
python app.py
```

### ⚡ **COMANDOS PRINCIPALES**

```bash
# INICIO RÁPIDO
python quick_start.py                    # Inicio completo
python quick_start.py --verify-only      # Solo verificar
python quick_start.py --test-quick       # Ejecutar tests rápidos
python quick_start.py --force-download   # Forzar descarga modelos

# TESTING
python -m backend.tests.run_all_tests              # Todos los tests
python -m backend.tests.run_all_tests --quick      # Tests rápidos
python -m backend.tests.run_all_tests --module mmpose  # Solo MMPose

# UTILIDADES
python verify_system.py             # Verificar dependencias
python download_models.py           # Solo descargar modelos
python app.py                       # Solo servidor (manual)
```

### 🔗 **ENDPOINTS API PRINCIPALES**

```bash
# GESTIÓN DE SESIONES
POST http://localhost:5000/api/session/start
GET  http://localhost:5000/api/session/status
POST http://localhost:5000/api/session/cancel

# PROCESAMIENTO
POST http://localhost:5000/api/chunks/receive
POST http://localhost:5000/api/cameras/recalibrate

# UTILIDADES
GET  http://localhost:5000/health
```

## 🧩 COMPONENTES CLAVE A ENTENDER

### 1️⃣ **app.py - Servidor Flask Principal**

```python
# Funciones principales:
start_session()           # Inicia nueva sesión de grabación
receive_chunk()          # Recibe y procesa chunk de video
cancel_session()         # Cancela sesión y limpia datos
recalibrate_cameras()    # Recalibra parámetros extrínsecos

# Variables globales importantes:
current_session = None   # Sesión actual activa
data_manager = None      # Gestor de datos
```

### 2️⃣ **ensemble_processor.py - Fusión de Modelos**

```python
# Proceso de ensemble:
fuse_coco_keypoints()         # Fusiona 17 keypoints COCO con pesos
extract_additional_keypoints() # Extrae keypoints únicos (pies)
process_frame_ensemble()      # Procesa frame completo con 4 modelos

# Pesos de fusión:
ENSEMBLE_WEIGHTS = {
    'hrnet_w48_coco': 0.6,    # Modelo más preciso
    'hrnet_w32_coco': 0.4,    # Complementario
    'resnet50_rle_coco': 1.0, # Keypoints únicos
    'wholebody_coco': 1.0     # Pies detallados
}
```

### 3️⃣ **triangulation.py - Reconstrucción 3D**

```python
# Algoritmo principal:
triangulate_dlt()             # Direct Linear Transform
_triangulate_point_dlt()      # Triangulación de punto individual
compute_reprojection_error()  # Validación por reproyección

# Parámetros críticos:
REPROJECTION_THRESHOLD = 5.0  # Máx error en píxeles
MIN_CAMERAS = 2               # Mínimo 2 cámaras para triangular
```

### 4️⃣ **pose_detector.py - Wrapper MMPose**

```python
# Clase principal:
class MMPoseInferencerWrapper:
    initialize_models()       # Carga 4 modelos en GPUs
    inference_batch()         # Procesa lote de frames
    cleanup()                # Libera memoria GPU

# Configuración GPU:
PRIMARY_GPU = 'cuda:0'       # HRNet models
SECONDARY_GPU = 'cuda:1'     # ResNet + WholeBody
```

## ⚠️ INCONGRUENCIAS CORREGIDAS

1. **✅ Fixed**: `quick_start.py` ahora muestra mejor info sobre modelos encontrados
2. **✅ Fixed**: Añadidas opciones `--test` y `--test-quick` al quick_start
3. **✅ Verified**: Todas las rutas de archivos son consistentes
4. **✅ Verified**: Los imports entre módulos son correctos
5. **✅ Verified**: La configuración de GPU es consistente en todo el sistema

## 🎯 PRÓXIMOS PASOS RECOMENDADOS

### Para empezar a desarrollar:

1. **Leer README.md principal** - Visión general
2. **Ejecutar `python quick_start.py --test-quick`** - Verificar que todo funciona
3. **Revisar `app.py`** - Entender la API
4. **Estudiar `triangulation.py`** - El corazón del sistema
5. **Experimentar con tests** - `backend/tests/`

### Para usar en producción:

1. **`python verify_system.py`** - Verificar dependencias
2. **`python download_models.py`** - Descargar modelos
3. **`python quick_start.py`** - Iniciar sistema completo
4. **Conectar cliente Code** - Para enviar videos

---

**🦴 Sistema de 5,800+ líneas listo para análisis de gonartrosis con máxima precisión.**

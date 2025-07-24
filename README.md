# 🦴 Server - Sistema de Análisis de Marcha para Detección de Gonartrosis

**Backend de procesamiento de video con reconstrucción 3D de keypoints**

## 📋 Descripción del Proyecto

Sistema de análisis de marcha diseñado para detectar patrones de gonartrosis mediante:

- **Procesamiento multi-cámara**: Recepción y procesamiento de chunks de video de 3-5 cámaras Orbbec Gemini 335L
- **Detección de pose 2D**: Ensemble learning con 4 modelos MMPose para máxima precisión
- **Reconstrucción 3D**: Triangulación optimizada de keypoints sin ground truth
- **Análisis cinemático**: Evaluación triplanar de rodilla (flexión-extensión, varo-valgo, rotación tibial)

## 🏗️ Arquitectura del Sistema

### Pipeline de Procesamiento

```
Chunks Video (Client) → Flask API → Extracción Frames → MMPose (4 modelos) 
                                         ↓
Reconstrucción 3D ← Ensemble Learning ← Keypoints 2D (ponderados)
       ↓
Análisis Cinemático → Resultados Clínicos
```

### Modelos MMPose Utilizados

#### **Modelos COCO (17 keypoints) - Máxima Precisión:**
1. **HRNet-W48**: `td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288` (GPU:0, peso: 0.6)
2. **HRNet-W32**: `td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192` (GPU:0, peso: 0.4)

#### **Modelos Extendidos (keypoints adicionales piernas):**
3. **ResNet50-RLE**: `td-hm_res50_rle-8xb64-210e_coco-256x192` (GPU:1)
4. **COCO-WholeBody**: `wholebody_2d_keypoint_topdown_coco-wholebody` (GPU:1, incluye 6 puntos por pie)

## 🚀 Instalación y Configuración

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar MMPose Models

Los modelos se descargan automáticamente en la primera ejecución, pero para máxima estabilidad:

```bash
# Crear directorios
mkdir -p mmpose_models/configs/pose2d
mkdir -p mmpose_models/checkpoints

# Los archivos .py y .pth se descargarán automáticamente
```

### 3. Configurar GPUs

Editar `config/settings.py`:

```python
class ProcessingConfig:
    primary_gpu: str = 'cuda:0'    # HRNet models
    secondary_gpu: str = 'cuda:1'  # ResNet + WholeBody
```

### 4. Ejecutar Servidor

```bash
python app.py
```

El servidor estará disponible en: `http://localhost:5000`

## 📡 API Endpoints

### Gestión de Sesiones

- **POST** `/api/session/start` - Iniciar nueva sesión
- **GET** `/api/session/status` - Estado de sesión actual  
- **POST** `/api/session/cancel` - Cancelar sesión y limpiar datos

### Procesamiento de Video

- **POST** `/api/chunks/receive` - Recibir chunks de video para procesamiento
- **POST** `/api/cameras/recalibrate` - Recalibrar parámetros extrínsecos

### Utilidades

- **GET** `/health` - Health check del servidor

## 📁 Estructura de Datos

```
data/
├── unprocessed/                     # Videos recibidos del cliente
│   └── patient{X}/
│       └── session{Y}/
│           └── camera{Z}/
│               ├── 1.mp4, 2.mp4...  # Chunks de video
│               └── 2D/              # Keypoints sin procesar
│                   ├── points/
│                   │   ├── hrnet_w48/    # 1.npy, 2.npy...
│                   │   ├── hrnet_w32/
│                   │   ├── resnet50_rle/
│                   │   └── wholebody/
│                   └── confidence/
│                       └── [same structure]
│
└── processed/                       # Datos procesados
    ├── photos_from_video/          # Frames extraídos (temporal)
    ├── 2D_keypoints/               # Keypoints finales (ensemble)
    │   └── patient{X}/session{Y}/camera{Z}/
    │       ├── 1.npy, 2.npy...     # Coordenadas 2D finales
    │       └── 1_confidence.npy... # Confianzas finales
    └── 3D_keypoints/               # Reconstrucción 3D
        └── patient{X}/session{Y}/
            ├── 1.npy, 2.npy...     # Puntos 3D triangulados
            ├── 1_confidence.npy... # Confianzas 3D
            └── 1_info.npz...       # Metadatos reconstrucción
```

## 🎯 Ensemble Learning Strategy

### Fusión de Keypoints COCO (17 puntos)
```python
# Promedio ponderado por confianza
final_weight = model_weight * confidence_score
fused_keypoint = Σ(final_weight * keypoint) / Σ(final_weight)
```

### Keypoints Adicionales de Piernas
- **WholeBody**: 6 puntos por pie (dedos, talón, centro)
- **ResNet50**: Procesamiento especializado para mayor robustez
- **Fusión**: Promedio cuando múltiples modelos detectan el mismo punto

### Filtrado por Confianza
- Threshold mínimo: 0.3
- Puntos únicos: Sin ponderación cuando solo un modelo los detecta

## 🏗️ Calibración de Cámaras

### Parámetros Intrínsecos
- **Método**: Tablero de ajedrez (9x6 esquinas, 25mm cuadros)
- **Almacenamiento**: `config/camera_intrinsics.py`
- **Actualización**: Automática tras calibración exitosa

### Parámetros Extrínsecos  
- **Método**: Estimación por correspondencias de keypoints 2D multi-frame
- **Referencia**: Cámara 0 como origen del sistema de coordenadas
- **Recalibración**: Botón en frontend para recalcular tras mover cámaras

## 🔺 Reconstrucción 3D

### Métodos de Triangulación

#### 1. Direct Linear Transform (DLT) - Principal
```python
# Sistema Ax = 0 para cada punto
# Optimización: SVD para solución de mínimos cuadrados
point_3d = triangulate_dlt(cameras, keypoints_2d)
```

#### 2. OpenCV Triangulation - Alternativo  
```python
# Para pares de cámaras
points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
```

### Validación por Error de Reproyección
- **Threshold**: 5.0 píxeles máximo
- **Método**: Error RMS entre proyección y observación
- **Filtrado**: Descarte automático de puntos con alto error

## 🧪 Sistema de Testing

### Ejecución de Tests

```bash
# Test completo del sistema
python -m backend.tests.test_reconstruction

# Test rápido
python -c "from backend.tests.test_reconstruction import run_quick_test; print(run_quick_test())"
```

### Tests Incluidos

1. **Datos Sintéticos**: Esqueleto humano 3D → proyección → triangulación
2. **Pipeline Completo**: Validación round-trip 3D→2D→3D  
3. **Datos Reales**: Testing con sesiones de pacientes procesadas
4. **Calibración**: Verificación de parámetros intrínsecos/extrínsecos

### Métricas de Calidad

- **RMSE 3D**: < 1cm (excelente), < 5cm (bueno)
- **Error Reproyección**: < 2 píxeles (óptimo)  
- **Round-trip Error**: < 1μm (sin pérdida de información)

## ⚙️ Configuración Avanzada

### Optimización de GPU

```python
# config/settings.py
class ProcessingConfig:
    # Distribución de modelos
    primary_gpu: str = 'cuda:0'      # HRNet (más pesados)
    secondary_gpu: str = 'cuda:1'    # ResNet + WholeBody
    
    # Parámetros de procesamiento  
    target_fps: int = 15             # FPS extracción frames
    confidence_threshold: float = 0.3 # Filtro confianza mínima
```

### Pesos del Ensemble

```python
ensemble_weights = {
    'hrnet_w48_coco': 0.6,    # Modelo más preciso
    'hrnet_w32_coco': 0.4,    # Complementario
    'resnet50_rle_coco': 1.0, # Keypoints únicos
    'wholebody_coco': 1.0     # Pies detallados
}
```

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. Error MMPose Import
```bash
# Reinstalar MMPose
pip uninstall mmpose mmcv mmdet mmengine
pip install mmpose==1.3.1 mmcv==2.1.0 mmdet==3.2.0 mmengine==0.10.1
```

#### 2. GPU Memory Error
```python
# Reducir batch size en config/settings.py
ProcessingConfig.target_fps = 10  # Menos frames por chunk
```

#### 3. Calibración Falla
```python
# Verificar keypoints 2D
python -c "
from backend.reconstruction.calibration import calibration_system
status = calibration_system.get_calibration_status()
print(status)
"
```

#### 4. Alto Error de Reproyección
- Verificar sincronización entre cámaras
- Recalibrar parámetros extrínsecos  
- Revisar calidad de keypoints 2D de entrada

## 📊 Métricas de Rendimiento

### Tiempos de Procesamiento (GPU RTX 3080)

- **Extracción frames**: ~2s por chunk (5s video)
- **MMPose inference**: ~0.1s por frame por modelo
- **Ensemble learning**: ~0.01s por frame  
- **Triangulación 3D**: ~0.001s por frame
- **Total pipeline**: ~8s por chunk (4 modelos, 15 fps)

### Precisión Esperada

- **Keypoints 2D**: ±1-2 píxeles (después de ensemble)
- **Reconstrucción 3D**: ±5-10mm (dependiente de setup cámaras)
- **Error reproyección**: <2 píxeles (sistema bien calibrado)

## 🤝 Integración con Cliente

### Flujo de Comunicación

1. **Cliente** inicia sesión → **POST** `/api/session/start`
2. **Cliente** envía chunks → **POST** `/api/chunks/receive` 
3. **Servidor** procesa automáticamente → Keypoints 2D → Ensemble → 3D
4. **Cliente** cancela si necesario → **POST** `/api/session/cancel`

### Sincronización

- Chunks procesados **conforme llegan** (no batch)
- **Numeración global** de frames para sincronización multi-cámara
- **Limpieza automática** de archivos temporales

## 📈 Roadmap de Desarrollo

### Fase Actual: ✅ Reconstrucción 3D Base
- [x] Pipeline MMPose completo
- [x] Ensemble learning 4 modelos  
- [x] Triangulación DLT optimizada
- [x] Sistema de calibración automática
- [x] Tests de validación completos

### Próxima Fase: 🔄 Análisis Cinemático
- [ ] Cálculo de ángulos articulares 3D
- [ ] Análisis triplanar de rodilla
- [ ] Detección de patrones de gonartrosis  
- [ ] Dashboard de resultados clínicos

### Fase Futura: 🚀 Optimización y Escalabilidad
- [ ] Procesamiento asíncrono con Celery
- [ ] Base de datos para histórico de pacientes
- [ ] API REST completa para integración clínica
- [ ] Optimización de inferencia (TensorRT/ONNX)

## 📄 Licencia

Proyecto de investigación - Universidad [PENDIENTE]

## 👥 Equipo de Desarrollo

- **Análisis de Marcha**: Especialistas en biomecánica
- **Computer Vision**: Implementación MMPose + reconstrucción 3D  
- **Backend**: Sistema distribuido Flask + procesamiento GPU
- **Investigación**: Validación clínica de métricas de gonartrosis

---

**� Nota**: Este es un sistema de investigación enfocado en máxima precisión. Cada error de medición cuenta para el análisis clínico final.

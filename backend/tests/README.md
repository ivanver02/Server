# 🧪 Sistema de Testing - Análisis de Marcha

**Suite completa de tests para validar todos los componentes del sistema**

## 📋 Descripción

Este directorio contiene tests exhaustivos para validar cada componente del sistema de análisis de marcha:

- **MMPose Integration**: Tests de modelos de pose, ensemble learning, y distribución GPU
- **Camera Calibration**: Tests de calibración intrínseca/extrínseca y validación de parámetros
- **3D Triangulation**: Tests de reconstrucción 3D, DLT, y error de reproyección
- **Full Pipeline**: Tests de integración del pipeline completo end-to-end

## 🚀 Uso Rápido

### Ejecutar Tests Rápidos (Sin dependencias pesadas)
```bash
# Tests básicos sin MMPose/GPU
python -m backend.tests.run_all_tests --quick
```

### Ejecutar Suite Completa
```bash
# Todos los tests incluyendo dependencias
python -m backend.tests.run_all_tests
```

### Tests por Módulo
```bash
# Solo MMPose
python -m backend.tests.run_all_tests --module mmpose

# Solo calibración
python -m backend.tests.run_all_tests --module calibration

# Solo triangulación  
python -m backend.tests.run_all_tests --module triangulation

# Solo reconstrucción completa
python -m backend.tests.run_all_tests --module reconstruction
```

## 📁 Estructura de Tests

```
backend/tests/
├── __init__.py                    # Inicialización del paquete
├── run_all_tests.py              # 🎯 Test runner principal
├── test_mmpose.py                 # Tests de MMPose y ensemble
├── test_calibration.py            # Tests de calibración de cámaras
├── test_triangulation.py          # Tests de triangulación 3D
├── test_reconstruction.py         # Tests del pipeline completo
└── README.md                      # Esta documentación
```

## 🧪 Descripción de Tests

### 1. **test_mmpose.py** - Tests de MMPose
Valida el sistema de detección de poses y ensemble learning:

```python
# Tests incluidos:
- TestMMPoseSingleModel       # Inicialización e inferencia individual
- TestEnsembleLearning        # Fusión ponderada de 4 modelos
- TestGPUDistribution         # Distribución cuda:0 y cuda:1

# Ejecución individual:
python -c "from backend.tests.test_mmpose import run_quick_mmpose_test; run_quick_mmpose_test()"
```

**Componentes validados:**
- ✅ Inicialización correcta de modelos MMPose
- ✅ Inferencia en imagen sintética de figura humana
- ✅ Ensemble learning con pesos (HRNet-W48: 0.6, HRNet-W32: 0.4)
- ✅ Filtrado por confianza mínima (threshold: 0.3)
- ✅ Distribución de modelos en 2 GPUs

### 2. **test_calibration.py** - Tests de Calibración
Valida el sistema de calibración de cámaras:

```python
# Tests incluidos:
- TestCameraIntrinsics        # Matriz K, distorsión, tablero ajedrez
- TestExtrinsicCalibration    # Poses de cámaras, R|t válidas
- TestReprojectionError       # Error de reproyección para validación

# Ejecución individual:
python -c "from backend.tests.test_calibration import test_quick_calibration; test_quick_calibration()"
```

**Componentes validados:**
- ✅ Generación de puntos de tablero de ajedrez (9x6, 25mm)
- ✅ Matrices intrínsecas válidas (fx, fy, cx, cy)
- ✅ Parámetros de distorsión en rangos realistas
- ✅ Matrices de rotación ortogonales (det=1)
- ✅ Vectores de traslación con separación mínima
- ✅ Consistencia de proyección 3D→2D

### 3. **test_triangulation.py** - Tests de Triangulación 3D
Valida la reconstrucción 3D mediante triangulación:

```python
# Tests incluidos:
- TestDLTTriangulation         # Direct Linear Transform
- TestReprojectionValidation   # Validación por error reproyección
- TestTriangulationEdgeCases   # Casos extremos (muy cerca/lejos)

# Ejecución individual:
python -c "from backend.tests.test_triangulation import test_quick_triangulation; test_quick_triangulation()"
```

**Componentes validados:**
- ✅ Triangulación perfecta sin ruido (error < 1e-6 mm)
- ✅ Triangulación con ruido realista (1 píxel → <50mm error)
- ✅ Validación de geometría (buena vs mala separación cámaras)
- ✅ Error de reproyección < 5 píxeles
- ✅ Casos edge: puntos muy cercanos/lejanos/infinito

### 4. **test_reconstruction.py** - Tests de Pipeline Completo
Valida el sistema completo de reconstrucción:

```python
# Tests incluidos:
- Datos sintéticos             # Esqueleto humano 3D → 2D → 3D
- Pipeline completo            # Validación round-trip
- Datos reales                 # Testing con sesiones de pacientes
- Calibración automática       # Verificación sistema calibración

# Ejecución individual:
python -c "from backend.tests.test_reconstruction import run_quick_test; run_quick_test()"
```

**Componentes validados:**
- ✅ Generación de esqueleto humano sintético (17 keypoints COCO)
- ✅ Sistema de cámaras multi-vista (3-5 cámaras)
- ✅ Round-trip validation 3D→2D→3D
- ✅ Métricas de calidad (RMSE, reproyección, round-trip error)
- ✅ Integración con sistema de calibración

## 📊 Métricas de Calidad

### Thresholds de Éxito

| Métrica | Excelente | Bueno | Aceptable | Fallo |
|---------|-----------|-------|-----------|-------|
| **RMSE 3D** | < 1cm | < 5cm | < 10cm | > 10cm |
| **Error Reproyección** | < 2px | < 5px | < 10px | > 10px |
| **Round-trip Error** | < 1μm | < 1mm | < 5mm | > 5mm |
| **Confianza Promedio** | > 0.8 | > 0.5 | > 0.3 | < 0.3 |

### Ejemplo de Output Exitoso
```
🧪 EJECUTANDO TESTS TRIANGULACIÓN
==================================================
test_perfect_triangulation ... OK
test_noisy_triangulation ... OK
test_triangulation_geometry_validation ... OK

📊 RESUMEN TESTS TRIANGULACIÓN
==================================================
✅ Tests exitosos: 8
❌ Tests fallidos: 0
💥 Errores: 0
```

## 🔧 Configuración de Tests

### Variables de Entorno
```bash
# Opcional: Configurar nivel de logging
export PYTHONPATH=.
export LOG_LEVEL=INFO

# Para tests sin GPU
export CUDA_VISIBLE_DEVICES=""
```

### Dependencias para Tests Completos
```bash
# Mínimas (tests rápidos)
pip install numpy scipy

# Completas (todos los tests)
pip install numpy scipy opencv-python torch mmpose mmcv mmdet mmengine
```

### Configuración de GPU
```python
# config/settings.py
class ProcessingConfig:
    primary_gpu: str = 'cuda:0'      # Tests usan ambas GPUs
    secondary_gpu: str = 'cuda:1'    # si están disponibles
```

## 🚨 Troubleshooting

### Errores Comunes

#### 1. Import Error MMPose
```bash
# Error: ModuleNotFoundError: No module named 'mmpose'
# Solución:
pip install mmpose==1.3.1 mmcv==2.1.0 mmdet==3.2.0 mmengine==0.10.1
```

#### 2. CUDA Out of Memory
```bash
# Error: CUDA out of memory
# Solución: Ejecutar solo tests rápidos
python -m backend.tests.run_all_tests --quick
```

#### 3. OpenCV Import Error
```bash
# Error: No module named 'cv2'
# Solución:
pip install opencv-python
```

#### 4. Test Fallidos por Precisión Numérica
```python
# Error: AssertionError: Arrays are not almost equal
# Solución: Ajustar tolerancias en tests
np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
```

### Debugging Tests

#### Ejecutar con Información Detallada
```bash
# Verbose output
python -m backend.tests.run_all_tests --verbose

# Solo un test específico
python -m unittest backend.tests.test_triangulation.TestDLTTriangulation.test_perfect_triangulation -v
```

#### Verificar Estado del Sistema
```python
# En Python interactivo
from backend.tests.run_all_tests import run_quick_tests
results = run_quick_tests()
print(results)
```

## 📈 Interpretación de Resultados

### Status Codes
- **0**: Todos los tests pasaron
- **1**: Algunos tests fallaron  
- **130**: Tests interrumpidos por usuario (Ctrl+C)

### Niveles de Funcionalidad del Sistema

#### 🟢 SISTEMA OPERATIVO (0 fallos)
- Todos los componentes funcionan correctamente
- Listo para producción
- Proceder con descarga de modelos y inicio del servidor

#### 🟡 SISTEMA FUNCIONAL (1 fallo)
- Funcionalidad principal intacta
- Limitaciones menores en componentes secundarios
- Revisar warnings pero puede usarse

#### 🔴 SISTEMA REQUIERE ATENCIÓN (2+ fallos)
- Múltiples componentes tienen problemas
- Revisión necesaria antes de usar
- Verificar instalación de dependencias

## 💡 Mejores Prácticas

### Desarrollo
- **Ejecutar tests rápidos** antes de cada commit
- **Tests completos** antes de release
- **Test específico** al desarrollar nuevo componente

### Integración Continua
```bash
# Script para CI/CD
#!/bin/bash
echo "Ejecutando tests del sistema..."
python -m backend.tests.run_all_tests

if [ $? -eq 0 ]; then
    echo "✅ Tests pasaron - Construyendo imagen Docker"
    docker build -t marcha-analysis-server .
else
    echo "❌ Tests fallaron - Deteniendo build"
    exit 1
fi
```

### Validación Pre-Deploy
```bash
# Checklist antes de deployment
python -m backend.tests.run_all_tests              # Tests completos
python verify_system.py                            # Verificar dependencias  
python download_models.py                          # Descargar modelos
python app.py &                                    # Test servidor
curl http://localhost:5000/health                  # Verificar API
```

---

**🎯 Objetivo**: Garantizar máxima precisión en el análisis de gonartrosis mediante testing exhaustivo de cada componente del sistema de reconstrucción 3D.

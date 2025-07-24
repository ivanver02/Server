@echo off
REM Script para crear y configurar entorno virtual para el proyecto Server
REM Ejecutar desde la carpeta Server

echo 🦴 CONFIGURACION ENTORNO VIRTUAL - SISTEMA ANALISIS DE MARCHA
echo =================================================================

REM Verificar que estamos en la carpeta correcta
if not exist "app.py" (
    echo ❌ Error: Este script debe ejecutarse desde la carpeta Server
    echo    Archivo app.py no encontrado
    pause
    exit /b 1
)

echo 📁 Directorio actual: %CD%
echo 🐍 Verificando Python...

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado o no está en PATH
    echo    Instala Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo ✅ Python disponible

REM Verificar si ya existe el entorno virtual
if exist "venv" (
    echo ⚠️  El entorno virtual 'venv' ya existe
    set /p choice="¿Quieres recrearlo? (s/N): "
    if /i "!choice!"=="s" (
        echo 🗑️ Eliminando entorno virtual existente...
        rmdir /s /q venv
    ) else (
        echo ✋ Usando entorno virtual existente
        goto activate_env
    )
)

echo 🔨 Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo ❌ Error: No se pudo crear el entorno virtual
    echo    Verifica que tienes permisos de escritura
    pause
    exit /b 1
)
echo ✅ Entorno virtual creado en: %CD%\venv

:activate_env
echo 🔄 Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Error: No se pudo activar el entorno virtual
    pause
    exit /b 1
)

echo ✅ Entorno virtual activado
echo 📋 Python del entorno: 
where python
python --version

echo 📦 Actualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ⚠️ Warning: No se pudo actualizar pip, continuando...
)

echo 📥 Instalando dependencias básicas...
python -m pip install numpy opencv-python scipy matplotlib pillow requests
if errorlevel 1 (
    echo ❌ Error instalando dependencias básicas
    pause
    exit /b 1
)

echo 🔥 Instalando PyTorch con CUDA...
echo    Esto puede tomar varios minutos...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ❌ Error instalando PyTorch con CUDA
    echo 🔄 Intentando instalar PyTorch CPU...
    python -m pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo ❌ Error instalando PyTorch
        pause
        exit /b 1
    )
)

echo 🧠 Instalando dependencias de MMPose...
echo    Esto puede tomar bastante tiempo...
python -m pip install mmengine
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
python -m pip install mmdet
python -m pip install mmpose
if errorlevel 1 (
    echo ❌ Error instalando MMPose stack
    echo 🔄 Intentando instalación alternativa...
    python -m pip install mmcv-full mmdetection mmpose --no-deps
)

echo 🌐 Instalando Flask y dependencias web...
python -m pip install flask flask-cors gunicorn
if errorlevel 1 (
    echo ❌ Error instalando Flask
    pause
    exit /b 1
)

echo 📊 Instalando dependencias adicionales...
python -m pip install tqdm pathlib2 pyyaml
if errorlevel 1 (
    echo ⚠️ Warning: Error con dependencias adicionales, continuando...
)

echo 🧪 Verificando instalación...
echo Verificando PyTorch...
python -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch no se importa correctamente
) else (
    echo ✅ PyTorch funcionando
)

echo Verificando OpenCV...
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ OpenCV no se importa correctamente
) else (
    echo ✅ OpenCV funcionando
)

echo Verificando MMPose...
python -c "import mmpose; print('✅ MMPose disponible')" 2>nul
if errorlevel 1 (
    echo ❌ MMPose no se importa correctamente
    echo    Es posible que necesites instalación manual
) else (
    echo ✅ MMPose funcionando
)

echo Verificando Flask...
python -c "import flask; print(f'✅ Flask {flask.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ Flask no se importa correctamente
) else (
    echo ✅ Flask funcionando
)

echo.
echo 🎉 INSTALACION COMPLETADA
echo =========================
echo.
echo 📋 Para usar el entorno virtual:
echo    1. Activar: venv\Scripts\activate.bat
echo    2. Desactivar: deactivate
echo.
echo 🚀 Próximos pasos:
echo    1. python verify_system.py      # Verificar sistema
echo    2. python download_models.py    # Descargar modelos MMPose
echo    3. python quick_start.py        # Iniciar sistema
echo.
echo 💡 El entorno virtual está activado en esta sesión
echo    Puedes ejecutar comandos Python directamente
echo.

REM Mantener la ventana abierta con el entorno activado
echo Presiona cualquier tecla para continuar con el entorno activado...
pause >nul
cmd /k

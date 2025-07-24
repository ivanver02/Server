"""
Configuración del Entorno de Testing
====================================

Script para verificar y configurar el entorno necesario
para ejecutar los tests independientes de procesamiento de video.
"""

import sys
import subprocess
import importlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dependencias requeridas para los tests
REQUIRED_PACKAGES = {
    'opencv-python': 'cv2',
    'numpy': 'numpy',
    'torch': 'torch',
    'mmcv': 'mmcv',
    'mmpose': 'mmpose'
}

# Dependencias básicas (siempre requeridas)
BASIC_PACKAGES = {
    'opencv-python': 'cv2',
    'numpy': 'numpy'
}

# Dependencias avanzadas (para test completo con MMPose)
ADVANCED_PACKAGES = {
    'torch': 'torch',
    'mmcv': 'mmcv', 
    'mmpose': 'mmpose'
}

def check_python_version():
    """Verificar versión de Python"""
    logger.info("🐍 Verificando versión de Python...")
    
    version = sys.version_info
    logger.info(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Se requiere Python 3.8 o superior")
        return False
    
    logger.info("✅ Versión de Python compatible")
    return True

def check_package(package_name: str, import_name: str) -> bool:
    """
    Verificar si un paquete está instalado
    
    Args:
        package_name: Nombre del paquete para pip
        import_name: Nombre para import en Python
        
    Returns:
        True si el paquete está disponible
    """
    try:
        module = importlib.import_module(import_name)
        
        # Obtener versión si está disponible
        version = getattr(module, '__version__', 'unknown')
        logger.info(f"   ✅ {package_name}: v{version}")
        return True
        
    except ImportError:
        logger.warning(f"   ❌ {package_name}: No instalado")
        return False

def install_package(package_name: str) -> bool:
    """
    Instalar un paquete usando pip
    
    Args:
        package_name: Nombre del paquete a instalar
        
    Returns:
        True si la instalación fue exitosa
    """
    try:
        logger.info(f"📦 Instalando {package_name}...")
        
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ {package_name} instalado correctamente")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error instalando {package_name}: {e}")
        logger.error(f"   stdout: {e.stdout}")
        logger.error(f"   stderr: {e.stderr}")
        return False

def check_cuda_availability():
    """Verificar disponibilidad de CUDA"""
    logger.info("🎮 Verificando CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA disponible con {device_count} GPU(s)")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"   GPU {i}: {gpu_name}")
            
            return True
        else:
            logger.warning("⚠️ CUDA no disponible, se usará CPU")
            return False
            
    except ImportError:
        logger.warning("⚠️ PyTorch no instalado, no se puede verificar CUDA")
        return False

def check_mmpose_models():
    """Verificar disponibilidad de modelos MMPose"""
    logger.info("🤖 Verificando modelos MMPose...")
    
    # Ruta a los modelos en el proyecto principal
    models_base_path = Path(__file__).parent.parent.parent.parent / 'mmpose_models'
    configs_path = models_base_path / 'configs' / 'pose2d'
    checkpoints_path = models_base_path / 'checkpoints'
    
    logger.info(f"   Buscando en: {models_base_path}")
    
    if not models_base_path.exists():
        logger.error("❌ Directorio mmpose_models no encontrado")
        logger.info("💡 Ejecuta: python download_models.py desde el directorio Server")
        return False
    
    # Verificar estructura
    if not configs_path.exists():
        logger.error("❌ Directorio configs no encontrado")
        return False
    
    if not checkpoints_path.exists():
        logger.error("❌ Directorio checkpoints no encontrado")
        return False
    
    # Contar archivos
    config_files = list(configs_path.glob("*.py"))
    checkpoint_files = list(checkpoints_path.glob("*.pth"))
    
    logger.info(f"   Configs encontrados: {len(config_files)}")
    logger.info(f"   Checkpoints encontrados: {len(checkpoint_files)}")
    
    if len(config_files) == 0 or len(checkpoint_files) == 0:
        logger.warning("⚠️ Modelos incompletos")
        logger.info("💡 Ejecuta: python download_models.py desde el directorio Server")
        return False
    
    logger.info("✅ Modelos MMPose disponibles")
    return True

def check_test_videos():
    """Verificar que existen los videos de test"""
    logger.info("📹 Verificando videos de test...")
    
    test_dir = Path(__file__).parent
    video_paths = {
        'camera0': test_dir / 'camera0' / '0.mp4',
        'camera1': test_dir / 'camera1' / '0.mp4',
        'camera2': test_dir / 'camera2' / '0.mp4'
    }
    
    missing_videos = []
    total_size_mb = 0
    
    for camera_id, video_path in video_paths.items():
        if video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            logger.info(f"   ✅ {camera_id}: {size_mb:.1f} MB")
        else:
            missing_videos.append(camera_id)
            logger.error(f"   ❌ {camera_id}: {video_path}")
    
    if missing_videos:
        logger.error(f"❌ Videos faltantes: {missing_videos}")
        logger.info("💡 Copia videos de prueba a las carpetas camera0/, camera1/, camera2/")
        return False
    
    logger.info(f"✅ Todos los videos disponibles ({total_size_mb:.1f} MB total)")
    return True

def setup_basic_environment():
    """Configurar entorno básico (OpenCV + NumPy)"""
    logger.info("\n🔧 CONFIGURACIÓN BÁSICA")
    logger.info("=" * 40)
    
    success = True
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Verificar/instalar paquetes básicos
    logger.info("\n📦 Verificando dependencias básicas...")
    
    for package_name, import_name in BASIC_PACKAGES.items():
        if not check_package(package_name, import_name):
            if not install_package(package_name):
                success = False
    
    return success

def setup_advanced_environment():
    """Configurar entorno avanzado (PyTorch + MMPose)"""
    logger.info("\n🚀 CONFIGURACIÓN AVANZADA")
    logger.info("=" * 40)
    
    success = True
    
    # Verificar/instalar paquetes avanzados
    logger.info("\n📦 Verificando dependencias avanzadas...")
    
    for package_name, import_name in ADVANCED_PACKAGES.items():
        if not check_package(package_name, import_name):
            logger.warning(f"⚠️ {package_name} no está instalado")
            logger.info(f"💡 Para instalarlo manualmente: pip install {package_name}")
            success = False
    
    # Verificar CUDA si PyTorch está disponible
    check_cuda_availability()
    
    # Verificar modelos MMPose
    if not check_mmpose_models():
        success = False
    
    return success

def run_environment_check():
    """Ejecutar verificación completa del entorno"""
    logger.info("🔍 VERIFICACIÓN COMPLETA DEL ENTORNO")
    logger.info("=" * 50)
    
    # Verificar entorno básico
    basic_ok = setup_basic_environment()
    
    # Verificar entorno avanzado
    advanced_ok = setup_advanced_environment()
    
    # Verificar videos de test
    logger.info("\n📹 VERIFICACIÓN DE VIDEOS")
    logger.info("=" * 30)
    videos_ok = check_test_videos()
    
    # Resumen final
    logger.info("\n" + "=" * 50)
    logger.info("📊 RESUMEN DE VERIFICACIÓN")
    logger.info("=" * 50)
    
    logger.info(f"✅ Entorno básico (OpenCV): {'OK' if basic_ok else 'FALTA'}")
    logger.info(f"🚀 Entorno avanzado (MMPose): {'OK' if advanced_ok else 'FALTA'}")
    logger.info(f"📹 Videos de test: {'OK' if videos_ok else 'FALTA'}")
    
    if basic_ok and videos_ok:
        logger.info("\n🎉 LISTO PARA TEST BÁSICO")
        logger.info("   Ejecuta: python simple_video_test.py")
    
    if basic_ok and advanced_ok and videos_ok:
        logger.info("\n🎯 LISTO PARA TEST COMPLETO")
        logger.info("   Ejecuta: python independent_test_processor.py")
    
    if not (basic_ok and videos_ok):
        logger.error("\n❌ CONFIGURACIÓN INCOMPLETA")
        logger.info("💡 Sigue las instrucciones mostradas arriba")
    
    return basic_ok and videos_ok

if __name__ == "__main__":
    success = run_environment_check()
    sys.exit(0 if success else 1)

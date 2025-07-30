"""
Script de verificación de archivos de configuración MMPose
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from config import mmpose_config

def verify_detector_files():
    """Verificar que todos los archivos de configuración y pesos existen"""
    
    print("🔍 Verificando archivos de detectores MMPose...")
    print(f"📁 Directorio base: {mmpose_config.models_dir}")
    
    detectors = ['vitpose', 'mspn', 'hrnet', 'csp']
    
    for detector_name in detectors:
        print(f"\n📋 Verificando {detector_name.upper()}:")
        
        config = getattr(mmpose_config, detector_name)
        
        # Verificar archivo de configuración
        config_path = mmpose_config.models_dir / config['pose2d']
        if config_path.exists():
            print(f"   ✅ Config: {config['pose2d']}")
        else:
            print(f"   ❌ Config: {config['pose2d']} - NO ENCONTRADO")
        
        # Verificar archivo de pesos
        weights_path = mmpose_config.models_dir / config['pose2d_weights']
        if weights_path.exists():
            print(f"   ✅ Weights: {config['pose2d_weights']}")
        else:
            print(f"   ❌ Weights: {config['pose2d_weights']} - NO ENCONTRADO")
    
    print("\n📂 Archivos disponibles en configs/pose2d/:")
    configs_dir = mmpose_config.models_dir / "configs" / "pose2d"
    if configs_dir.exists():
        for file in configs_dir.iterdir():
            if file.suffix == '.py':
                print(f"   📄 {file.name}")
    
    print("\n💾 Archivos disponibles en checkpoints/:")
    checkpoints_dir = mmpose_config.models_dir / "checkpoints"
    if checkpoints_dir.exists():
        for file in checkpoints_dir.iterdir():
            if file.suffix == '.pth':
                print(f"   🏋️  {file.name}")

if __name__ == "__main__":
    verify_detector_files()

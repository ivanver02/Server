#!/usr/bin/env python3
"""
Script de verificación completa del sistema de análisis de marcha.

Verifica:
- Dependencias de Python
- Modelos MMPose
- GPUs disponibles
- Configuración del sistema
- Tests básicos

Uso:
    python verify_system.py
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

class SystemVerifier:
    """Verificador completo del sistema"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.info = []
    
    def log_error(self, msg: str):
        """Registra un error"""
        self.errors.append(msg)
        print(f"❌ {msg}")
    
    def log_warning(self, msg: str):
        """Registra una advertencia"""
        self.warnings.append(msg)
        print(f"⚠️  {msg}")
    
    def log_info(self, msg: str):
        """Registra información"""
        self.info.append(msg)
        print(f"ℹ️  {msg}")
    
    def log_success(self, msg: str):
        """Registra éxito"""
        print(f"✅ {msg}")
    
    def check_python_version(self) -> bool:
        """Verifica versión de Python"""
        print("\n🐍 Verificando Python...")
        
        version = sys.version_info
        print(f"   Versión: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.log_error(f"Python 3.8+ requerido, encontrado {version.major}.{version.minor}")
            return False
        
        self.log_success(f"Python {version.major}.{version.minor}.{version.micro} OK")
        return True
    
    def check_dependencies(self) -> bool:
        """Verifica dependencias de Python"""
        print("\n📦 Verificando dependencias...")
        
        # Dependencias críticas
        critical_deps = [
            ('flask', 'Flask'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy'),
            ('requests', 'Requests')
        ]
        
        # Dependencias opcionales (GPUs/MMPose)
        optional_deps = [
            ('torch', 'PyTorch'),
            ('mmpose', 'MMPose'),
            ('mmcv', 'MMCV'),
            ('mmdet', 'MMDetection'),
            ('mmengine', 'MMEngine')
        ]
        
        all_ok = True
        
        # Verificar críticas
        for module_name, display_name in critical_deps:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.log_error(f"{display_name} no instalado")
                    all_ok = False
                else:
                    # Intentar importar para verificar
                    try:
                        importlib.import_module(module_name)
                        self.log_success(f"{display_name} OK")
                    except Exception as e:
                        self.log_error(f"{display_name} error al importar: {e}")
                        all_ok = False
            except Exception as e:
                self.log_error(f"Error verificando {display_name}: {e}")
                all_ok = False
        
        # Verificar opcionales
        print("\n   Dependencias opcionales:")
        for module_name, display_name in optional_deps:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.log_warning(f"{display_name} no instalado (necesario para inferencia)")
                else:
                    try:
                        importlib.import_module(module_name)
                        self.log_success(f"{display_name} OK")
                    except Exception as e:
                        self.log_warning(f"{display_name} error al importar: {e}")
            except Exception as e:
                self.log_warning(f"Error verificando {display_name}: {e}")
        
        return all_ok
    
    def check_gpu_support(self) -> Dict[str, bool]:
        """Verifica soporte GPU"""
        print("\n🎮 Verificando GPUs...")
        
        gpu_info = {
            'torch_available': False,
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_names': []
        }
        
        try:
            import torch
            gpu_info['torch_available'] = True
            self.log_success("PyTorch disponible")
            
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                self.log_success(f"CUDA disponible con {gpu_info['gpu_count']} GPU(s)")
                
                for i in range(gpu_info['gpu_count']):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_info['gpu_names'].append(gpu_name)
                    print(f"   GPU {i}: {gpu_name}")
                
            else:
                self.log_warning("CUDA no disponible - solo CPU")
        
        except ImportError:
            self.log_warning("PyTorch no instalado - no se puede verificar GPU")
        except Exception as e:
            self.log_error(f"Error verificando GPU: {e}")
        
        return gpu_info
    
    def check_mmpose_models(self) -> Dict[str, bool]:
        """Verifica modelos MMPose"""
        print("\n🤖 Verificando modelos MMPose...")
        
        models_path = self.base_path / "mmpose_models"
        configs_path = models_path / "configs" / "pose2d"
        checkpoints_path = models_path / "checkpoints"
        
        if not models_path.exists():
            self.log_warning("Directorio mmpose_models no existe")
            self.log_info("Ejecuta: python download_models.py")
            return {}
        
        # Modelos esperados
        expected_models = {
            'hrnet_w48_coco': {
                'config': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py',
                'checkpoint': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-1d86a0de_20220909.pth'
            },
            'hrnet_w32_coco': {
                'config': 'td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192.py',
                'checkpoint': 'td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192-81c58e40_20220909.pth'
            },
            'resnet50_rle_coco': {
                'config': 'td-hm_res50_rle-8xb64-210e_coco-256x192.py',
                'checkpoint': 'td-hm_res50_rle-8xb64-210e_coco-256x192-c4aa2b08_20220913.pth'
            },
            'wholebody_coco': {
                'config': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288.py',
                'checkpoint': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288-ce11e65b_20220913.pth'
            }
        }
        
        models_status = {}
        
        for model_name, files in expected_models.items():
            config_path = configs_path / files['config']
            checkpoint_path = checkpoints_path / files['checkpoint']
            
            config_ok = config_path.exists() and config_path.stat().st_size > 1024
            checkpoint_ok = checkpoint_path.exists() and checkpoint_path.stat().st_size > 1024*1024
            
            models_status[model_name] = config_ok and checkpoint_ok
            
            if models_status[model_name]:
                self.log_success(f"Modelo {model_name} OK")
            else:
                missing = []
                if not config_ok:
                    missing.append("config")
                if not checkpoint_ok:
                    missing.append("checkpoint")
                self.log_warning(f"Modelo {model_name} incompleto: falta {', '.join(missing)}")
        
        if not any(models_status.values()):
            self.log_warning("Ningún modelo MMPose encontrado")
            self.log_info("Ejecuta: python download_models.py")
        
        return models_status
    
    def check_directory_structure(self) -> bool:
        """Verifica estructura de directorios"""
        print("\n📁 Verificando estructura de directorios...")
        
        required_dirs = [
            "config",
            "backend",
            "backend/api",
            "backend/processing", 
            "backend/reconstruction",
            "backend/data_management",
            "backend/tests",
            "data",
            "data/unprocessed",
            "data/processed"
        ]
        
        optional_dirs = [
            "mmpose_models",
            "mmpose_models/configs",
            "mmpose_models/checkpoints",
            "logs"
        ]
        
        all_ok = True
        
        # Verificar directorios requeridos
        for dir_path in required_dirs:
            full_path = self.base_path / dir_path
            if full_path.exists():
                self.log_success(f"Directorio {dir_path}/ OK")
            else:
                self.log_error(f"Directorio requerido {dir_path}/ no existe")
                all_ok = False
        
        # Verificar directorios opcionales
        print("\n   Directorios opcionales:")
        for dir_path in optional_dirs:
            full_path = self.base_path / dir_path
            if full_path.exists():
                self.log_success(f"Directorio {dir_path}/ OK")
            else:
                self.log_warning(f"Directorio {dir_path}/ no existe")
        
        return all_ok
    
    def check_config_files(self) -> bool:
        """Verifica archivos de configuración"""
        print("\n⚙️  Verificando configuración...")
        
        config_files = [
            "config/settings.py",
            "config/camera_intrinsics.py", 
            "config/keypoint_mappings.py"
        ]
        
        all_ok = True
        
        for config_file in config_files:
            file_path = self.base_path / config_file
            if file_path.exists():
                self.log_success(f"Config {config_file} OK")
                
                # Verificar importación
                try:
                    if config_file == "config/settings.py":
                        sys.path.insert(0, str(self.base_path))
                        from config.settings import ServerConfig, ProcessingConfig
                        self.log_info(f"   Configuración cargada correctamente")
                except Exception as e:
                    self.log_warning(f"   Error cargando {config_file}: {e}")
            else:
                self.log_error(f"Config {config_file} no existe")
                all_ok = False
        
        return all_ok
    
    def run_basic_tests(self) -> bool:
        """Ejecuta tests básicos del sistema"""
        print("\n🧪 Ejecutando tests básicos...")
        
        try:
            # Test de importación de módulos principales
            sys.path.insert(0, str(self.base_path))
            
            # Test 1: Importar configuración
            try:
                from config.settings import ServerConfig, ProcessingConfig
                self.log_success("Config importada OK")
            except Exception as e:
                self.log_error(f"Error importando config: {e}")
                return False
            
            # Test 2: Importar gestión de datos
            try:
                from backend.data_management.data_manager import DataManager
                self.log_success("DataManager importado OK")
            except Exception as e:
                self.log_error(f"Error importando DataManager: {e}")
                return False
            
            # Test 3: Test de triangulación (si scipy disponible)
            try:
                from backend.tests.test_reconstruction import run_quick_test
                result = run_quick_test()
                if result:
                    self.log_success("Test de reconstrucción 3D OK")
                else:
                    self.log_warning("Test de reconstrucción falló")
            except Exception as e:
                self.log_warning(f"No se pudo ejecutar test 3D: {e}")
            
            return True
        
        except Exception as e:
            self.log_error(f"Error en tests básicos: {e}")
            return False
    
    def print_summary(self, gpu_info: Dict, models_status: Dict):
        """Imprime resumen final"""
        print("\n" + "="*70)
        print("📊 RESUMEN DEL SISTEMA")
        print("="*70)
        
        # Estado general
        if self.errors:
            print("❌ ESTADO: ERRORES CRÍTICOS")
            print("\n🚨 Errores que deben resolverse:")
            for error in self.errors:
                print(f"   • {error}")
        elif self.warnings:
            print("⚠️  ESTADO: LISTO CON ADVERTENCIAS")
        else:
            print("✅ ESTADO: SISTEMA COMPLETAMENTE LISTO")
        
        # Información del sistema
        print(f"\n🖥️  Sistema:")
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • Plataforma: {sys.platform}")
        
        # GPUs
        if gpu_info['cuda_available']:
            print(f"   • GPUs: {gpu_info['gpu_count']} CUDA disponibles")
            for i, name in enumerate(gpu_info['gpu_names']):
                print(f"     - GPU {i}: {name}")
        else:
            print("   • GPUs: Solo CPU (sin CUDA)")
        
        # Modelos
        models_ready = sum(models_status.values())
        models_total = len(models_status)
        print(f"   • Modelos MMPose: {models_ready}/{models_total} listos")
        
        # Próximos pasos
        print(f"\n🚀 Próximos pasos:")
        
        if self.errors:
            print("   1. Resolver errores críticos listados arriba")
            print("   2. Ejecutar: pip install -r requirements.txt")
            print("   3. Volver a ejecutar: python verify_system.py")
        elif models_ready < models_total:
            print("   1. Descargar modelos: python download_models.py")
            print("   2. Verificar nuevamente: python verify_system.py")
            print("   3. Ejecutar servidor: python app.py")
        else:
            print("   1. ¡Listo! Ejecutar servidor: python app.py")
            print("   2. Abrir en navegador: http://localhost:5000")
            if not gpu_info['cuda_available']:
                print("   3. Nota: Sin GPU, procesamiento será más lento")
        
        # Advertencias
        if self.warnings:
            print(f"\n⚠️  Advertencias ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
    
    def run_verification(self) -> bool:
        """Ejecuta verificación completa"""
        print("🦴 Verificador del Sistema de Análisis de Marcha")
        print("=" * 70)
        
        # Verificaciones paso a paso
        python_ok = self.check_python_version()
        deps_ok = self.check_dependencies()
        gpu_info = self.check_gpu_support()
        models_status = self.check_mmpose_models()
        dirs_ok = self.check_directory_structure()
        config_ok = self.check_config_files()
        tests_ok = self.run_basic_tests()
        
        # Resumen
        self.print_summary(gpu_info, models_status)
        
        # Retornar estado general
        return python_ok and deps_ok and dirs_ok and config_ok and not self.errors

def main():
    """Función principal"""
    verifier = SystemVerifier()
    success = verifier.run_verification()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

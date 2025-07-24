#!/usr/bin/env python3
"""
Script para descargar automáticamente los modelos MMPose necesarios
para el sistema de análisis de marcha.

Uso:
    python download_models.py

Los modelos se descargan en mmpose_models/ y el sistema los usa automáticamente.
"""

import os
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Configuración de modelos a descargar
MODELS_CONFIG = {
    # Modelos COCO principales (17 keypoints)
    'hrnet_w48_coco': {
        'config_url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.1/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py',
        'checkpoint_url': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-1d86a0de_20220909.pth',
        'config_file': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py',
        'checkpoint_file': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-1d86a0de_20220909.pth',
        'description': 'HRNet-W48 COCO - Modelo principal de máxima precisión'
    },
    
    'hrnet_w32_coco': {
        'config_url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.1/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192.py',
        'checkpoint_url': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192-81c58e40_20220909.pth',
        'config_file': 'td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192.py',
        'checkpoint_file': 'td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192-81c58e40_20220909.pth',
        'description': 'HRNet-W32 COCO - Modelo complementario'
    },
    
    'resnet50_rle_coco': {
        'config_url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.1/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_rle-8xb64-210e_coco-256x192.py',
        'checkpoint_url': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_rle-8xb64-210e_coco-256x192-c4aa2b08_20220913.pth',
        'config_file': 'td-hm_res50_rle-8xb64-210e_coco-256x192.py',
        'checkpoint_file': 'td-hm_res50_rle-8xb64-210e_coco-256x192-c4aa2b08_20220913.pth',
        'description': 'ResNet50-RLE COCO - Modelo robusto'
    },
    
    # Modelo WholeBody para keypoints adicionales
    'wholebody_coco': {
        'config_url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.1/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288.py',
        'checkpoint_url': 'https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288-ce11e65b_20220913.pth',
        'config_file': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288.py',
        'checkpoint_file': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288-ce11e65b_20220913.pth',
        'description': 'WholeBody COCO - Keypoints de cuerpo completo incluye pies'
    }
}

class ModelDownloader:
    """Descargador automático de modelos MMPose"""
    
    def __init__(self, base_path: str = "mmpose_models"):
        self.base_path = Path(base_path)
        self.configs_path = self.base_path / "configs" / "pose2d"
        self.checkpoints_path = self.base_path / "checkpoints"
        
        # Crear directorios si no existen
        self.configs_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, destination: Path, description: str = "") -> bool:
        """
        Descarga un archivo desde URL con barra de progreso
        
        Args:
            url: URL del archivo a descargar
            destination: Ruta de destino
            description: Descripción para mostrar
            
        Returns:
            bool: True si descarga exitosa
        """
        try:
            print(f"📥 Descargando {description}...")
            print(f"   URL: {url}")
            print(f"   Destino: {destination}")
            
            # Si el archivo ya existe, preguntar si reemplazar
            if destination.exists():
                print(f"   ⚠️  El archivo ya existe: {destination.name}")
                response = input("   ¿Reemplazar? (y/N): ").lower().strip()
                if response != 'y':
                    print("   ⏭️  Saltando descarga")
                    return True
            
            # Descargar con requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Obtener tamaño total
            total_size = int(response.headers.get('content-length', 0))
            
            # Escribir archivo con barra de progreso
            downloaded = 0
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Mostrar progreso cada MB
                        if downloaded % (1024*1024) == 0:
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"   📊 Progreso: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({percent:.1f}%)")
                            else:
                                print(f"   📊 Descargado: {downloaded/(1024*1024):.1f}MB")
            
            print(f"   ✅ Descarga completada: {destination.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error descargando {url}: {e}")
            if destination.exists():
                destination.unlink()  # Eliminar archivo parcial
            return False
    
    def verify_file(self, file_path: Path, expected_size_min: int = 1024) -> bool:
        """
        Verifica que un archivo descargado sea válido
        
        Args:
            file_path: Ruta del archivo
            expected_size_min: Tamaño mínimo esperado en bytes
            
        Returns:
            bool: True si el archivo es válido
        """
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        if file_size < expected_size_min:
            print(f"   ⚠️  Archivo muy pequeño: {file_size} bytes < {expected_size_min}")
            return False
        
        return True
    
    def download_model(self, model_name: str, config: Dict) -> bool:
        """
        Descarga un modelo completo (config + checkpoint)
        
        Args:
            model_name: Nombre del modelo
            config: Configuración del modelo
            
        Returns:
            bool: True si descarga exitosa
        """
        print(f"\n🚀 Procesando modelo: {model_name}")
        print(f"   📝 {config['description']}")
        
        success = True
        
        # Descargar archivo de configuración
        config_dest = self.configs_path / config['config_file']
        if not self.download_file(
            config['config_url'], 
            config_dest, 
            f"Config {model_name}"
        ):
            success = False
        
        # Descargar checkpoint
        checkpoint_dest = self.checkpoints_path / config['checkpoint_file']
        if not self.download_file(
            config['checkpoint_url'], 
            checkpoint_dest, 
            f"Checkpoint {model_name}"
        ):
            success = False
        
        # Verificar archivos
        if success:
            if not self.verify_file(config_dest, 1024):  # Config mínimo 1KB
                print(f"   ❌ Config inválido: {config_dest}")
                success = False
            
            if not self.verify_file(checkpoint_dest, 1024*1024):  # Checkpoint mínimo 1MB
                print(f"   ❌ Checkpoint inválido: {checkpoint_dest}")
                success = False
        
        if success:
            print(f"   ✅ Modelo {model_name} descargado correctamente")
        else:
            print(f"   ❌ Error descargando modelo {model_name}")
        
        return success
    
    def download_all_models(self) -> Dict[str, bool]:
        """
        Descarga todos los modelos configurados
        
        Returns:
            Dict[str, bool]: Estado de descarga por modelo
        """
        print("🔄 Iniciando descarga de modelos MMPose...")
        print(f"📁 Directorio base: {self.base_path.absolute()}")
        
        results = {}
        
        for model_name, config in MODELS_CONFIG.items():
            results[model_name] = self.download_model(model_name, config)
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Imprime resumen de descargas"""
        print("\n" + "="*60)
        print("📊 RESUMEN DE DESCARGAS")
        print("="*60)
        
        successful = []
        failed = []
        
        for model_name, success in results.items():
            if success:
                successful.append(model_name)
                print(f"✅ {model_name}: OK")
            else:
                failed.append(model_name)
                print(f"❌ {model_name}: FALLÓ")
        
        print(f"\n📈 Total: {len(successful)}/{len(results)} modelos descargados")
        
        if failed:
            print(f"\n⚠️  Modelos fallidos: {', '.join(failed)}")
            print("   Puedes volver a ejecutar el script para reintentarlos")
        
        if successful:
            print(f"\n🎉 Modelos listos para usar:")
            for model in successful:
                print(f"   • {model}: {MODELS_CONFIG[model]['description']}")
    
    def check_existing_models(self) -> Dict[str, bool]:
        """Verifica qué modelos ya están descargados"""
        print("🔍 Verificando modelos existentes...")
        
        existing = {}
        
        for model_name, config in MODELS_CONFIG.items():
            config_path = self.configs_path / config['config_file']
            checkpoint_path = self.checkpoints_path / config['checkpoint_file']
            
            config_ok = self.verify_file(config_path, 1024)
            checkpoint_ok = self.verify_file(checkpoint_path, 1024*1024)
            
            existing[model_name] = config_ok and checkpoint_ok
            
            if existing[model_name]:
                print(f"   ✅ {model_name}: Ya descargado")
            else:
                missing = []
                if not config_ok:
                    missing.append("config")
                if not checkpoint_ok:
                    missing.append("checkpoint")
                print(f"   ❌ {model_name}: Falta {', '.join(missing)}")
        
        return existing

def main():
    """Función principal"""
    print("🦴 Descargador de Modelos MMPose - Sistema de Análisis de Marcha")
    print("=" * 70)
    
    # Crear descargador
    downloader = ModelDownloader()
    
    # Verificar modelos existentes
    existing = downloader.check_existing_models()
    
    # Preguntar si continuar si algunos ya existen
    existing_count = sum(existing.values())
    total_count = len(existing)
    
    if existing_count > 0:
        print(f"\n📋 {existing_count}/{total_count} modelos ya descargados")
        if existing_count == total_count:
            print("🎉 ¡Todos los modelos ya están descargados!")
            response = input("¿Redescargar todos? (y/N): ").lower().strip()
            if response != 'y':
                print("✋ Saliendo sin cambios")
                return
    
    # Descargar modelos
    results = downloader.download_all_models()
    
    # Mostrar resumen
    downloader.print_summary(results)
    
    # Verificar si todo está listo
    all_successful = all(results.values())
    
    if all_successful:
        print("\n🚀 ¡Sistema listo! Puedes ejecutar:")
        print("   python app.py")
    else:
        print("\n⚠️  Algunos modelos fallaron. Verifica conexión a internet y vuelve a ejecutar")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

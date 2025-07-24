#!/usr/bin/env python3
"""
Script de inicio rápido para el sistema de análisis de marcha.

Este script:
1. Verifica el sistema
2. Descarga modelos si es necesario  
3. Inicia el servidor

Uso:
    python quick_start.py
    python quick_start.py --verify-only    # Solo verificación
    python quick_start.py --force-download # Forzar descarga modelos
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            [sys.executable] + command,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completado")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}:")
        print(f"   {e.stderr}")
        return False, e.stderr

def main():
    parser = argparse.ArgumentParser(description="Inicio rápido del sistema de análisis de marcha")
    parser.add_argument('--verify-only', action='store_true', help='Solo verificar sistema')
    parser.add_argument('--force-download', action='store_true', help='Forzar descarga de modelos')
    parser.add_argument('--skip-verification', action='store_true', help='Saltar verificación inicial')
    
    args = parser.parse_args()
    
    print("🦴 Inicio Rápido - Sistema de Análisis de Marcha")
    print("=" * 60)
    
    # Paso 1: Verificación del sistema (a menos que se salte)
    if not args.skip_verification:
        success, output = run_command(['verify_system.py'], 'Verificación del sistema')
        if not success:
            print("\n⚠️  La verificación detectó problemas.")
            print("   Revisa los errores arriba y resuelve antes de continuar.")
            
            # Preguntar si continuar de todos modos
            response = input("\n¿Continuar de todos modos? (y/N): ").lower().strip()
            if response != 'y':
                print("✋ Deteniendo inicio. Resuelve los problemas y vuelve a intentar.")
                return 1
    
    # Solo verificación?
    if args.verify_only:
        print("\n✅ Verificación completada. Usa 'python quick_start.py' para iniciar el servidor.")
        return 0
    
    # Paso 2: Descargar modelos si es necesario
    models_path = Path('mmpose_models/checkpoints')
    
    # Verificar si hay modelos
    has_models = False
    if models_path.exists():
        pth_files = list(models_path.glob('*.pth'))
        has_models = len(pth_files) >= 4  # Al menos 4 modelos
    
    if args.force_download or not has_models:
        print(f"\n📥 {'Forzando descarga' if args.force_download else 'Modelos no encontrados, descargando'}...")
        success, output = run_command(['download_models.py'], 'Descarga de modelos MMPose')
        if not success:
            print("\n❌ Error descargando modelos. Verifica conexión a internet.")
            return 1
    else:
        print("\n✅ Modelos MMPose ya disponibles")
    
    # Paso 3: Iniciar servidor
    print("\n🚀 Iniciando servidor Flask...")
    print("   URL: http://localhost:5000")
    print("   Presiona Ctrl+C para detener")
    print("   Logs aparecerán a continuación...")
    print("-" * 60)
    
    try:
        # Ejecutar servidor sin capturar output (para ver logs en tiempo real)
        result = subprocess.run([sys.executable, 'app.py'], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error iniciando servidor: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n✋ Servidor detenido por el usuario")
        return 0

if __name__ == "__main__":
    sys.exit(main())

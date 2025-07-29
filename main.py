#!/usr/bin/env python3
"""
Punto de entrada principal para el servidor de procesamiento de video
Sistema de análisis de marcha para detección de gonartrosis

Uso:
    python main.py

Este archivo permite ejecutar directamente el servidor Flask sin necesidad
de llamar a app.py directamente.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Función principal para iniciar el servidor"""
    try:
        print("=" * 60)
        print("🏥 SISTEMA DE ANÁLISIS DE MARCHA - DETECCIÓN DE GONARTROSIS")
        print("=" * 60)
        print(f"📁 Directorio del proyecto: {project_root}")
        print("🚀 Iniciando servidor de procesamiento...")
        print()
        
        # Importar y ejecutar la aplicación Flask
        from app import app, logger
        from config import server_config, data_config
        
        # Mostrar información de configuración
        logger.info(f"Servidor iniciando en: http://{server_config.host}:{server_config.port}")
        logger.info(f"Directorio de datos: {data_config.base_data_dir}")
        logger.info(f"Modo debug: {server_config.debug}")
        
        print(f"🌐 Servidor disponible en: http://{server_config.host}:{server_config.port}")
        print(f"📊 Health check: http://{server_config.host}:{server_config.port}/health")
        print()
        print("Presiona Ctrl+C para detener el servidor")
        print("-" * 60)
        
        # Ejecutar servidor Flask
        app.run(
            host=server_config.host,
            port=server_config.port,
            debug=server_config.debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
        sys.exit(0)
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("Verifica que todas las dependencias estén instaladas.")
        print("Ejecuta: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error crítico al iniciar el servidor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

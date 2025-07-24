#!/usr/bin/env python3
"""
Test básico de MMPose
"""
import os
import sys
from pathlib import Path

# Configurar rutas
current_dir = Path(__file__).parent
server_dir = current_dir.parent.parent
sys.path.insert(0, str(server_dir))

print("🚀 Iniciando test básico de MMPose...")

try:
    # Test 1: Importar MMPose
    print("📦 Importando MMPose...")
    from mmpose.apis import MMPoseInferencer
    print("✅ MMPose importado correctamente")
    
    # Test 2: Verificar modelos
    models_dir = server_dir / "mmpose_models"
    print(f"📁 Directorio de modelos: {models_dir}")
    print(f"📁 Existe: {models_dir.exists()}")
    
    if models_dir.exists():
        files = list(models_dir.iterdir())
        print(f"📋 Archivos encontrados: {len(files)}")
        for f in files:
            print(f"  • {f.name}")
    
    # Test 3: Intentar inicializar un modelo simple
    print("🔧 Intentando inicializar modelo...")
    
    # Buscar archivos de modelo
    config_files = list(models_dir.glob("*.py")) if models_dir.exists() else []
    checkpoint_files = list(models_dir.glob("*.pth")) if models_dir.exists() else []
    
    print(f"📋 Configs: {len(config_files)}")
    print(f"📋 Checkpoints: {len(checkpoint_files)}")
    
    if config_files and checkpoint_files:
        config = config_files[0]
        checkpoint = checkpoint_files[0]
        
        print(f"🎯 Usando config: {config.name}")
        print(f"🎯 Usando checkpoint: {checkpoint.name}")
        
        try:
            inferencer = MMPoseInferencer(
                pose2d=str(config),
                pose2d_weights=str(checkpoint),
                device='cpu'
            )
            print("✅ Modelo inicializado exitosamente")
            
            # Test 4: Crear imagen de prueba
            print("📸 Creando imagen de prueba...")
            import cv2
            import numpy as np
            
            # Crear imagen simple
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_img, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            test_img_path = current_dir / "test_image.jpg"
            cv2.imwrite(str(test_img_path), test_img)
            print(f"✅ Imagen creada: {test_img_path}")
            
            # Test 5: Procesar imagen
            print("🔄 Procesando imagen...")
            
            result_generator = inferencer(
                str(test_img_path),
                show=False,
                return_vis=True
            )
            
            results = next(result_generator)
            print("✅ Imagen procesada exitosamente")
            print(f"📊 Resultados: {type(results)}")
            
            if 'predictions' in results:
                predictions = results['predictions']
                print(f"📊 Predicciones: {len(predictions)}")
            
            # Limpiar
            if test_img_path.exists():
                test_img_path.unlink()
                print("🧹 Imagen de prueba eliminada")
            
        except Exception as e:
            print(f"❌ Error inicializando modelo: {e}")
    else:
        print("⚠️ No se encontraron archivos de modelo necesarios")
    
    print("🎉 Test completado")
    
except Exception as e:
    print(f"❌ Error en test: {e}")
    import traceback
    traceback.print_exc()

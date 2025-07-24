#!/usr/bin/env python3
"""
Test directo de modelos MMPose - Solo usando mi código y mis rutas
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Configurar rutas
current_dir = Path(__file__).parent
server_dir = current_dir.parent.parent
sys.path.insert(0, str(server_dir))

print("🎯 Test directo de modelos MMPose")
print("=" * 50)

# Definir rutas específicas de mis modelos
models_dir = server_dir / "mmpose_models"

# Configuración de modelos disponibles
AVAILABLE_MODELS = {
    "hrnet_w48": {
        "config": models_dir / "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
        "checkpoint": models_dir / "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth",
        "name": "HRNet-W48"
    },
    "vitpose_huge": {
        "config": models_dir / "td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py", 
        "checkpoint": models_dir / "td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth",
        "name": "ViTPose-Huge"
    }
}

def test_model_files():
    """Verificar que todos los archivos de modelo existen"""
    print("📁 Verificando archivos de modelos...")
    
    for model_key, model_info in AVAILABLE_MODELS.items():
        config_exists = model_info["config"].exists()
        checkpoint_exists = model_info["checkpoint"].exists()
        
        print(f"📋 {model_info['name']}:")
        print(f"   Config: {'✅' if config_exists else '❌'} {model_info['config'].name}")
        print(f"   Checkpoint: {'✅' if checkpoint_exists else '❌'} {model_info['checkpoint'].name}")
        
        if not config_exists or not checkpoint_exists:
            print(f"   ⚠️ Archivos faltantes para {model_info['name']}")
            return False
    
    print("✅ Todos los archivos de modelo encontrados")
    return True

def create_test_image():
    """Crear imagen de prueba con una persona simulada"""
    print("🖼️ Creando imagen de prueba...")
    
    # Crear imagen con fondo
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Dibujar una figura humana simple
    # Cabeza
    cv2.circle(img, (320, 120), 30, (255, 255, 255), -1)
    
    # Cuerpo
    cv2.rectangle(img, (300, 150), (340, 280), (255, 255, 255), -1)
    
    # Brazos
    cv2.rectangle(img, (260, 160), (300, 180), (255, 255, 255), -1)  # Brazo izquierdo
    cv2.rectangle(img, (340, 160), (380, 180), (255, 255, 255), -1)  # Brazo derecho
    
    # Piernas
    cv2.rectangle(img, (305, 280), (320, 380), (255, 255, 255), -1)  # Pierna izquierda
    cv2.rectangle(img, (325, 280), (340, 380), (255, 255, 255), -1)  # Pierna derecha
    
    # Agregar texto
    cv2.putText(img, "TEST PERSON", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    test_img_path = current_dir / "test_person.jpg"
    cv2.imwrite(str(test_img_path), img)
    
    print(f"✅ Imagen creada: {test_img_path}")
    return test_img_path

def test_model_inference(model_key):
    """Probar inferencia con un modelo específico"""
    try:
        from mmpose.apis import MMPoseInferencer
        
        model_info = AVAILABLE_MODELS[model_key]
        print(f"\n🤖 Probando {model_info['name']}...")
        
        # Inicializar inferencer con rutas directas
        inferencer = MMPoseInferencer(
            pose2d=str(model_info["config"]),
            pose2d_weights=str(model_info["checkpoint"]),
            device='cpu'
        )
        print(f"✅ {model_info['name']} inicializado correctamente")
        
        # Crear imagen de prueba
        test_img_path = create_test_image()
        
        # Realizar inferencia
        print(f"🔄 Ejecutando inferencia con {model_info['name']}...")
        result_generator = inferencer(
            str(test_img_path),
            show=False,
            return_vis=True
        )
        
        results = next(result_generator)
        print(f"✅ Inferencia completada")
        
        # Analizar resultados
        if 'predictions' in results:
            predictions = results['predictions']
            print(f"📊 Predicciones encontradas: {len(predictions)}")
            
            if len(predictions) > 0:
                first_pred = predictions[0]
                if 'keypoints' in first_pred:
                    keypoints = first_pred['keypoints']
                    print(f"🎯 Keypoints detectados: {len(keypoints)} puntos")
                    
                    # Mostrar algunos keypoints
                    for i, kp in enumerate(keypoints[:5]):  # Solo los primeros 5
                        x, y, conf = kp[0], kp[1], kp[2] if len(kp) > 2 else 1.0
                        print(f"   Punto {i}: ({x:.1f}, {y:.1f}) conf={conf:.3f}")
                else:
                    print("⚠️ Sin keypoints en la predicción")
        else:
            print("⚠️ Sin predicciones en los resultados")
        
        # Limpiar
        if test_img_path.exists():
            test_img_path.unlink()
            print("🧹 Imagen de prueba eliminada")
        
        return True
        
    except Exception as e:
        print(f"❌ Error con {model_info['name']}: {e}")
        return False

def main():
    """Función principal de testing"""
    print("🚀 Iniciando test directo de modelos...")
    
    # Test 1: Verificar archivos
    if not test_model_files():
        print("❌ Test fallido: archivos de modelo faltantes")
        return
    
    # Test 2: Importar MMPose
    try:
        from mmpose.apis import MMPoseInferencer
        print("✅ MMPose importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando MMPose: {e}")
        return
    
    # Test 3: Probar cada modelo
    successful_models = []
    failed_models = []
    
    for model_key in AVAILABLE_MODELS:
        if test_model_inference(model_key):
            successful_models.append(AVAILABLE_MODELS[model_key]['name'])
        else:
            failed_models.append(AVAILABLE_MODELS[model_key]['name'])
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN FINAL:")
    print(f"✅ Modelos exitosos: {len(successful_models)}")
    for model in successful_models:
        print(f"   • {model}")
    
    if failed_models:
        print(f"❌ Modelos fallidos: {len(failed_models)}")
        for model in failed_models:
            print(f"   • {model}")
    
    print("🎉 Test completado")

if __name__ == "__main__":
    main()

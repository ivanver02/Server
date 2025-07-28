"""
Test de detección de keypoints 2D con HRNet-W48
Procesa atleta.jpg y muestra coordenadas y confianzas por pantalla
"""
import sys
import os
import numpy as np
from pathlib import Path
import cv2
import logging

# Agregar el directorio padre al path para importar las configuraciones
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hrnet_keypoint_detection():
    """
    Test de detección de keypoints usando HRNet-W48
    Procesa la imagen atleta.jpg y muestra resultados por pantalla
    """
    try:
        # Importar configuraciones
        from config import mmpose_config
        from backend.processing.detectors.mmpose.hrnet_w48_detector import HRNetW48Detector
        
        print("=" * 60)
        print("TEST DE DETECCIÓN DE KEYPOINTS 2D - HRNet-W48")
        print("=" * 60)
        
        # Configuración del detector
        hrnet_config = mmpose_config.hrnet_w48
        print(f"Configuración HRNet-W48:")
        print(f"   • Modelo: {hrnet_config['model_name']}")
        print(f"   • Config: {hrnet_config['config']}")
        print(f"   • Checkpoint: {hrnet_config['checkpoint']}")
        print(f"   • Device: {hrnet_config['device']}")
        print()
        
        # Cargar imagen atleta.jpg del mismo directorio
        test_dir = Path(__file__).parent
        image_path = test_dir / "atleta.jpg"
        
        print(f"Procesando imagen: {image_path}")
        
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error cargando imagen: {image_path}")
            return False
        
        print(f"Dimensiones imagen: {image.shape[1]}x{image.shape[0]} pixels")
        print()
        
        # Inicializar detector HRNet-W48
        print("Inicializando detector HRNet-W48...")
        detector = HRNetW48Detector()
        
        if not detector.initialize():
            print("Error inicializando detector HRNet-W48")
            return False
        
        print(" Detector inicializado correctamente")
        print()
        
        # Detectar keypoints
        print("🔍 Detectando keypoints...")
        keypoints_2d, confidence_scores = detector.detect_frame(image)
        
        if keypoints_2d is None or confidence_scores is None:
            print(" No se detectaron keypoints en la imagen")
            return False
        
        print(f" Detección completada: {len(keypoints_2d)} keypoints detectados")
        print()
        
        # Mostrar resultados
        print(" RESULTADOS DE DETECCIÓN:")
        print("=" * 50)
        
        # Nombres de keypoints COCO (17 puntos)
        coco_keypoint_names = [
            "nose",           # 0
            "left_eye",       # 1
            "right_eye",      # 2
            "left_ear",       # 3
            "right_ear",      # 4
            "left_shoulder",  # 5
            "right_shoulder", # 6
            "left_elbow",     # 7
            "right_elbow",    # 8
            "left_wrist",     # 9
            "right_wrist",    # 10
            "left_hip",       # 11
            "right_hip",      # 12
            "left_knee",      # 13
            "right_knee",     # 14
            "left_ankle",     # 15
            "right_ankle"     # 16
        ]
        
        print("COORDENADAS DE KEYPOINTS:")
        print("-" * 50)
        for i, (keypoint, confidence) in enumerate(zip(keypoints_2d, confidence_scores)):
            name = coco_keypoint_names[i] if i < len(coco_keypoint_names) else f"keypoint_{i}"
            x, y = keypoint[0], keypoint[1]
            print(f"{i:2d}. {name:15} | x:{x:7.2f}, y:{y:7.2f} | conf:{confidence:.3f}")
        
        print()
        print("ESTADÍSTICAS DE CONFIANZA:")
        print("-" * 50)
        print(f"Confianza promedio: {np.mean(confidence_scores):.3f}")
        print(f"Confianza mínima:   {np.min(confidence_scores):.3f}")
        print(f"Confianza máxima:   {np.max(confidence_scores):.3f}")
        print(f"Desviación estándar: {np.std(confidence_scores):.3f}")
        
        # Filtrar keypoints por confianza
        high_confidence_threshold = 0.7
        high_conf_keypoints = confidence_scores >= high_confidence_threshold
        
        print()
        print(f"KEYPOINTS CON ALTA CONFIANZA (>= {high_confidence_threshold}):")
        print("-" * 50)
        for i, (keypoint, confidence) in enumerate(zip(keypoints_2d, confidence_scores)):
            if confidence >= high_confidence_threshold:
                name = coco_keypoint_names[i] if i < len(coco_keypoint_names) else f"keypoint_{i}"
                x, y = keypoint[0], keypoint[1]
                print(f"{i:2d}. {name:15} | x:{x:7.2f}, y:{y:7.2f} | conf:{confidence:.3f} ")
        
        print()
        print("ARRAYS NUMPY:")
        print("-" * 50)
        print("Keypoints 2D shape:", keypoints_2d.shape)
        print("Confidence scores shape:", confidence_scores.shape)
        print()
        print("Keypoints 2D array:")
        print(keypoints_2d)
        print()
        print("Confidence scores array:")
        print(confidence_scores)
        
        # Dibujar keypoints en la imagen y guardar
        print()
        print("💾 GUARDANDO IMAGEN CON KEYPOINTS...")
        print("-" * 50)
        
        # Crear copia de la imagen para dibujar
        image_with_keypoints = image.copy()
        
        # Colores para diferentes tipos de keypoints
        colors = {
            'face': (0, 255, 255),      # Amarillo para cara (nose, eyes, ears)
            'upper': (0, 255, 0),       # Verde para parte superior (shoulders, elbows, wrists)
            'lower': (255, 0, 0),       # Azul para parte inferior (hips, knees, ankles)
        }
        
        # Clasificar keypoints por tipo
        keypoint_types = [
            'face', 'face', 'face', 'face', 'face',  # 0-4: nose, eyes, ears
            'upper', 'upper', 'upper', 'upper', 'upper', 'upper',  # 5-10: shoulders, elbows, wrists
            'lower', 'lower', 'lower', 'lower', 'lower', 'lower'   # 11-16: hips, knees, ankles
        ]
        
        # Dibujar keypoints
        for i, (keypoint, confidence) in enumerate(zip(keypoints_2d, confidence_scores)):
            if confidence > 0.3:  # Solo dibujar keypoints con confianza mínima
                x, y = int(keypoint[0]), int(keypoint[1])
                keypoint_type = keypoint_types[i] if i < len(keypoint_types) else 'upper'
                color = colors[keypoint_type]
                
                # Dibujar círculo para el keypoint
                cv2.circle(image_with_keypoints, (x, y), 5, color, -1)
                
                # Dibujar número del keypoint
                cv2.putText(image_with_keypoints, str(i), (x + 8, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Dibujar conexiones entre keypoints (esqueleto)
        skeleton_connections = [
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),  # eyes to ears
            (5, 6),          # shoulders
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10), # right arm
            (5, 11), (6, 12), # shoulders to hips
            (11, 12),        # hips
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16)  # right leg
        ]
        
        for connection in skeleton_connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints_2d) and pt2_idx < len(keypoints_2d) and
                confidence_scores[pt1_idx] > 0.3 and confidence_scores[pt2_idx] > 0.3):
                
                pt1 = (int(keypoints_2d[pt1_idx][0]), int(keypoints_2d[pt1_idx][1]))
                pt2 = (int(keypoints_2d[pt2_idx][0]), int(keypoints_2d[pt2_idx][1]))
                cv2.line(image_with_keypoints, pt1, pt2, (255, 255, 255), 2)
        
        # Guardar imagen con keypoints
        output_path = test_dir / "atleta_con_keypoints.jpg"
        cv2.imwrite(str(output_path), image_with_keypoints)
        
        print(f"✅ Imagen guardada: {output_path}")
        print(f"   • Keypoints dibujados: {len([c for c in confidence_scores if c > 0.3])}")
        print(f"   • Colores: Amarillo=Cara, Verde=Torso/Brazos, Azul=Piernas")
        print(f"   • Conexiones del esqueleto incluidas")
        
        print()
        print("=" * 60)
        print(" TEST COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f" Error de importación: {e}")
        print("Asegúrate de que MMPose esté instalado y configurado correctamente")
        return False
    except Exception as e:
        print(f" Error durante el test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Iniciando test de detección de keypoints 2D...")
    print()
    
    # Ejecutar test
    success = test_hrnet_keypoint_detection()
    
    if success:
        print("\nTest ejecutado exitosamente")
    else:
        print("\n Test falló")
        sys.exit(1)

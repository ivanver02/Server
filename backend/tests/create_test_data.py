#!/usr/bin/env python3
"""
Crear datos de prueba simulados para test de reconstrucción 3D
"""

import numpy as np
from pathlib import Path
import sys

# Agregar el directorio del proyecto al path
sys.path.append(str(Path(__file__).parent.parent.parent))

def create_test_keypoints_data(base_data_dir: Path, patient_id: str = "1", session_id: str = "6"):
    """
    Crear datos simulados de keypoints 2D para testing
    """
    session_dir = base_data_dir / "processed" / "2D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    
    # Crear directorios
    for camera_id in [0, 1, 2]:
        camera_dir = session_dir / f"camera{camera_id}"
        coord_dir = camera_dir / "coordinates"
        conf_dir = camera_dir / "confidence"
        
        coord_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)
        
        # Simular keypoints para varios frames
        for frame_idx in range(5):  # 5 frames de prueba
            for chunk_idx in range(2):  # 2 chunks por frame
                frame_key = f"frame{frame_idx:03d}_{chunk_idx:02d}"
                
                # Generar keypoints simulados (17 keypoints de COCO)
                num_keypoints = 17
                
                # Keypoints base (persona centrada en la imagen)
                base_keypoints = np.array([
                    [320, 180],  # nose
                    [315, 170], [325, 170],  # eyes
                    [310, 175], [330, 175],  # ears
                    [300, 220], [340, 220],  # shoulders
                    [280, 280], [360, 280],  # elbows
                    [260, 340], [380, 340],  # wrists
                    [310, 300], [330, 300],  # hips
                    [300, 420], [340, 420],  # knees
                    [295, 540], [345, 540],  # ankles
                ], dtype=np.float32)
                
                # Agregar variación por cámara (simular diferentes perspectivas)
                camera_offset = camera_id * 20  # Offset horizontal
                camera_keypoints = base_keypoints.copy()
                camera_keypoints[:, 0] += camera_offset  # Desplazamiento X
                
                # Agregar ruido pequeño
                noise = np.random.normal(0, 2, camera_keypoints.shape)
                camera_keypoints += noise
                
                # Asegurar que están dentro de los límites de imagen
                camera_keypoints[:, 0] = np.clip(camera_keypoints[:, 0], 0, 640)
                camera_keypoints[:, 1] = np.clip(camera_keypoints[:, 1], 0, 480)
                
                # Generar confianzas (más altas para keypoints principales)
                confidences = np.random.uniform(0.4, 0.9, num_keypoints).astype(np.float32)
                
                # Algunos keypoints con baja confianza
                if np.random.random() < 0.3:
                    low_conf_idx = np.random.choice(num_keypoints, 2, replace=False)
                    confidences[low_conf_idx] = np.random.uniform(0.1, 0.3, 2)
                
                # Guardar archivos
                coord_file = coord_dir / f"{frame_key}.npy"
                conf_file = conf_dir / f"{frame_key}.npy"
                
                np.save(coord_file, camera_keypoints)
                np.save(conf_file, confidences)
        
        print(f"✓ Datos creados para cámara {camera_id}")
    
    print(f"✅ Datos de prueba creados en {session_dir}")
    return session_dir

if __name__ == "__main__":
    base_data_dir = Path(__file__).parent.parent.parent / "data"
    create_test_keypoints_data(base_data_dir)

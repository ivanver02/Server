#!/usr/bin/env python3
"""
Script de testing independiente para procesar videos con ViTPose-huge
Procesa videos reales de 3 cámaras usando el nuevo modelo ViTPose-huge
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any
import json
import time

# Añadir el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_mmpose():
    """Configurar MMPose para el testing"""
    try:
        from mmpose.apis import MMPoseInferencer
        logger.info("✅ MMPose importado correctamente")
        return True
    except ImportError as e:
        logger.error(f"❌ Error importando MMPose: {e}")
        return False

def extract_frames_from_videos():
    """Extraer frames de los videos para procesamiento"""
    videos_dir = Path(__file__).parent
    cameras = ['camera0', 'camera1', 'camera2']
    frames_extracted = {}
    
    for camera in cameras:
        camera_dir = videos_dir / camera
        if not camera_dir.exists():
            logger.warning(f"⚠️ Directorio {camera} no encontrado")
            continue
            
        # Buscar archivo de video
        video_files = list(camera_dir.glob("*.mp4"))
        if not video_files:
            logger.warning(f"⚠️ No se encontraron videos en {camera}")
            continue
            
        video_path = video_files[0]
        logger.info(f"📹 Procesando video: {video_path}")
        
        # Crear directorio para frames
        frames_dir = camera_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Extraer frames
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        extracted_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extraer 1 frame cada 30 (aproximadamente 1 por segundo)
            if frame_count % 30 == 0:
                frame_filename = frames_dir / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                extracted_frames.append(str(frame_filename))
                
                # Limitar a 10 frames por video para el test
                if len(extracted_frames) >= 10:
                    break
                    
            frame_count += 1
            
        cap.release()
        frames_extracted[camera] = extracted_frames
        logger.info(f"✅ {camera}: {len(extracted_frames)} frames extraídos")
    
    return frames_extracted

def test_vitpose_huge(frames_dict: Dict[str, List[str]]):
    """Probar el modelo ViTPose-huge con frames reales"""
    try:
        from mmpose.apis import MMPoseInferencer
        
        # Configurar rutas del modelo
        models_dir = Path(__file__).parent.parent.parent / "mmpose_models"
        config_path = models_dir / "td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
        checkpoint_path = models_dir / "td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
        
        if not config_path.exists():
            logger.error(f"❌ Config no encontrado: {config_path}")
            return False
            
        if not checkpoint_path.exists():
            logger.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
            return False
            
        logger.info("🚀 Inicializando ViTPose-huge...")
        
        # Verificar CUDA
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🎮 Usando dispositivo: {device}")
        
        # Crear inferencer
        inferencer = MMPoseInferencer(
            pose2d=str(config_path),
            pose2d_weights=str(checkpoint_path),
            device=device
        )
        
        logger.info("✅ ViTPose-huge inicializado correctamente")
        
        # Procesar frames de cada cámara
        results_all_cameras = {}
        
        for camera, frame_paths in frames_dict.items():
            logger.info(f"🔄 Procesando {camera} con {len(frame_paths)} frames")
            camera_results = []
            
            for i, frame_path in enumerate(frame_paths):
                start_time = time.time()
                
                # Realizar inferencia
                try:
                    results = inferencer(frame_path)
                    processing_time = time.time() - start_time
                    
                    # Procesar resultados
                    if results and 'predictions' in results and len(results['predictions']) > 0:
                        prediction = results['predictions'][0]  # Primera persona detectada
                        
                        keypoints = np.array(prediction.get('keypoints', []))
                        keypoint_scores = np.array(prediction.get('keypoint_scores', []))
                        bbox = prediction.get('bbox', None)
                        bbox_score = prediction.get('bbox_score', 0.0)
                        
                        result = {
                            'frame_index': i,
                            'frame_path': frame_path,
                            'keypoints': keypoints.tolist() if len(keypoints) > 0 else [],
                            'keypoint_scores': keypoint_scores.tolist() if len(keypoint_scores) > 0 else [],
                            'bbox': bbox,
                            'bbox_score': float(bbox_score),
                            'processing_time': processing_time,
                            'person_detected': True
                        }
                        
                        logger.info(f"  Frame {i+1}/{len(frame_paths)}: ✅ Persona detectada (score: {bbox_score:.3f}, tiempo: {processing_time:.3f}s)")
                        
                    else:
                        result = {
                            'frame_index': i,
                            'frame_path': frame_path,
                            'keypoints': [],
                            'keypoint_scores': [],
                            'bbox': None,
                            'bbox_score': 0.0,
                            'processing_time': processing_time,
                            'person_detected': False
                        }
                        
                        logger.warning(f"  Frame {i+1}/{len(frame_paths)}: ⚠️ No se detectó persona")
                    
                    camera_results.append(result)
                    
                except Exception as e:
                    logger.error(f"  Frame {i+1}/{len(frame_paths)}: ❌ Error: {e}")
                    continue
            
            results_all_cameras[camera] = camera_results
            
            # Estadísticas por cámara
            successful_detections = sum(1 for r in camera_results if r['person_detected'])
            avg_processing_time = np.mean([r['processing_time'] for r in camera_results])
            
            logger.info(f"📊 {camera}: {successful_detections}/{len(camera_results)} detecciones exitosas")
            logger.info(f"⏱️ {camera}: Tiempo promedio: {avg_processing_time:.3f}s por frame")
        
        # Guardar resultados
        results_file = Path(__file__).parent / "vitpose_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_all_cameras, f, indent=2)
        
        logger.info(f"💾 Resultados guardados en: {results_file}")
        
        # Estadísticas generales
        total_frames = sum(len(results) for results in results_all_cameras.values())
        total_detections = sum(
            sum(1 for r in results if r['person_detected'])
            for results in results_all_cameras.values()
        )
        
        logger.info(f"🎯 Resumen final:")
        logger.info(f"   Total frames procesados: {total_frames}")
        logger.info(f"   Total detecciones exitosas: {total_detections}")
        logger.info(f"   Tasa de éxito: {total_detections/total_frames*100:.1f}%" if total_frames > 0 else "N/A")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en testing de ViTPose-huge: {e}")
        return False

def main():
    """Función principal de testing"""
    logger.info("🧪 Iniciando test independiente de ViTPose-huge")
    
    # Verificar MMPose
    if not setup_mmpose():
        logger.error("❌ No se pudo configurar MMPose")
        return
    
    # Extraer frames de videos
    logger.info("🎬 Extrayendo frames de videos...")
    frames_dict = extract_frames_from_videos()
    
    if not frames_dict:
        logger.error("❌ No se encontraron videos para procesar")
        return
    
    total_frames = sum(len(frames) for frames in frames_dict.values())
    logger.info(f"📸 Total de frames extraídos: {total_frames}")
    
    # Probar ViTPose-huge
    logger.info("🤖 Iniciando pruebas con ViTPose-huge...")
    success = test_vitpose_huge(frames_dict)
    
    if success:
        logger.info("🎉 Test completado exitosamente!")
    else:
        logger.error("❌ Test falló")

if __name__ == "__main__":
    main()

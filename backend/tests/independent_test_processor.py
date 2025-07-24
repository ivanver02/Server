"""
Sistema de Testing Independiente para Procesamiento de Videos
=============================================================

Este módulo contiene toda la lógica de procesamiento duplicada del sistema principal
pero adaptada para funcionar de manera independiente dentro de la carpeta tests.

Funcionalidades:
- Procesamiento de videos de las 3 cámaras
- Extracción de frames
- Detección de poses con 4 modelos MMPose
- Generación de visualizaciones con keypoints
- Almacenamiento organizado de resultados

Todo se mantiene dentro de backend/tests/ sin afectar el sistema principal.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import os
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de rutas para el test
TEST_BASE_PATH = Path(__file__).parent
VIDEO_PATHS = {
    'camera0': TEST_BASE_PATH / 'camera0' / '0.mp4',
    'camera1': TEST_BASE_PATH / 'camera1' / '0.mp4', 
    'camera2': TEST_BASE_PATH / 'camera2' / '0.mp4'
}
PROCESSED_DATA_PATH = TEST_BASE_PATH / 'processed_data'
FRAMES_PATH = PROCESSED_DATA_PATH / 'frames'
KEYPOINTS_PATH = PROCESSED_DATA_PATH / 'keypoints'
VISUALIZATIONS_PATH = PROCESSED_DATA_PATH / 'visualizations'

# Configuración de modelos MMPose
MMPOSE_MODELS = {
    'hrnet_w48_coco': {
        'config': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288',
        'checkpoint': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-1d86a0de_20220909.pth',
        'keypoints': 17,
        'gpu': 'cuda:0',
        'weight': 0.6
    },
    'hrnet_w32_coco': {
        'config': 'td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192',
        'checkpoint': 'td-hm_hrnet-w32_udp-8xb32-210e_coco-256x192-81c58e40_20220909.pth',
        'keypoints': 17,
        'gpu': 'cuda:0', 
        'weight': 0.4
    },
    'resnet50_rle_coco': {
        'config': 'td-hm_res50_rle-8xb64-210e_coco-256x192',
        'checkpoint': 'td-hm_res50_rle-8xb64-210e_coco-256x192-c4aa2b08_20220913.pth',
        'keypoints': 17,
        'gpu': 'cuda:1',
        'weight': 1.0
    },
    'wholebody_coco': {
        'config': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288',
        'checkpoint': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-wholebody-384x288-ce11e65b_20220913.pth',
        'keypoints': 133,  # WholeBody tiene más keypoints
        'gpu': 'cuda:1',
        'weight': 1.0
    }
}

# Mapeo de keypoints COCO
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

@dataclass
class ProcessingConfig:
    """Configuración para el procesamiento de tests"""
    target_fps: int = 15
    max_frames: int = 150  # Máximo 10 segundos a 15fps
    confidence_threshold: float = 0.3
    primary_gpu: str = 'cuda:0'
    secondary_gpu: str = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'

class TestPhotoModel:
    """
    Modelo de foto duplicado para el sistema de testing
    Maneja metadatos y operaciones de una foto individual
    """
    
    def __init__(self, image_path: str, camera_id: int, frame_number: int):
        self.image_path = Path(image_path)
        self.camera_id = camera_id
        self.frame_number = frame_number
        self.timestamp = time.time()
        self.width = 0
        self.height = 0
        self.keypoints_data = {}
        self.processing_metadata = {}
        
        # Cargar imagen y obtener dimensiones
        self._load_image_info()
    
    def _load_image_info(self):
        """Cargar información básica de la imagen"""
        try:
            if self.image_path.exists():
                img = cv2.imread(str(self.image_path))
                if img is not None:
                    self.height, self.width = img.shape[:2]
                    logger.debug(f"Imagen cargada: {self.width}x{self.height}")
        except Exception as e:
            logger.error(f"Error cargando imagen {self.image_path}: {e}")
    
    def add_keypoints(self, model_name: str, keypoints: np.ndarray, 
                     confidence: np.ndarray, processing_time: float = 0.0):
        """Añadir keypoints detectados por un modelo"""
        self.keypoints_data[model_name] = {
            'keypoints': keypoints.copy(),
            'confidence': confidence.copy(),
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
        self.processing_metadata[model_name] = {
            'num_keypoints': len(keypoints),
            'avg_confidence': float(np.mean(confidence)) if len(confidence) > 0 else 0.0,
            'min_confidence': float(np.min(confidence)) if len(confidence) > 0 else 0.0,
            'max_confidence': float(np.max(confidence)) if len(confidence) > 0 else 0.0,
            'processing_time': processing_time
        }
    
    def get_keypoints(self, model_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Obtener keypoints de un modelo específico"""
        if model_name in self.keypoints_data:
            data = self.keypoints_data[model_name]
            return data['keypoints'], data['confidence']
        return None
    
    def get_all_models(self) -> List[str]:
        """Obtener lista de modelos que han procesado esta foto"""
        return list(self.keypoints_data.keys())
    
    def save_keypoints(self, output_dir: Path):
        """Guardar keypoints en archivos numpy"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, data in self.keypoints_data.items():
            # Guardar keypoints
            keypoints_file = output_dir / f"camera{self.camera_id}_frame{self.frame_number}_{model_name}_keypoints.npy"
            np.save(keypoints_file, data['keypoints'])
            
            # Guardar confianzas
            confidence_file = output_dir / f"camera{self.camera_id}_frame{self.frame_number}_{model_name}_confidence.npy"
            np.save(confidence_file, data['confidence'])
        
        # Guardar metadatos
        metadata_file = output_dir / f"camera{self.camera_id}_frame{self.frame_number}_metadata.json"
        metadata = {
            'camera_id': self.camera_id,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'image_path': str(self.image_path),
            'width': self.width,
            'height': self.height,
            'processing_metadata': self.processing_metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_visualization(self, output_dir: Path):
        """Crear visualización con keypoints superpuestos"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Cargar imagen original
            img = cv2.imread(str(self.image_path))
            if img is None:
                logger.error(f"No se pudo cargar imagen: {self.image_path}")
                return
            
            # Crear visualización para cada modelo
            for model_name in self.get_all_models():
                keypoints, confidence = self.get_keypoints(model_name)
                
                # Crear copia de la imagen
                vis_img = img.copy()
                
                # Dibujar keypoints
                self._draw_keypoints(vis_img, keypoints, confidence, model_name)
                
                # Guardar visualización
                output_file = output_dir / f"camera{self.camera_id}_frame{self.frame_number}_{model_name}_vis.jpg"
                cv2.imwrite(str(output_file), vis_img)
                
                logger.debug(f"Visualización guardada: {output_file}")
        
        except Exception as e:
            logger.error(f"Error creando visualización: {e}")
    
    def _draw_keypoints(self, img: np.ndarray, keypoints: np.ndarray, 
                       confidence: np.ndarray, model_name: str):
        """Dibujar keypoints en la imagen"""
        if len(keypoints) == 0:
            return
        
        # Colores para diferentes modelos
        colors = {
            'hrnet_w48_coco': (0, 255, 0),      # Verde
            'hrnet_w32_coco': (255, 0, 0),      # Azul
            'resnet50_rle_coco': (0, 0, 255),   # Rojo
            'wholebody_coco': (255, 255, 0)     # Cian
        }
        
        color = colors.get(model_name, (255, 255, 255))
        
        # Dibujar cada keypoint
        for i, (kp, conf) in enumerate(zip(keypoints, confidence)):
            if conf > 0.3:  # Solo dibujar si confianza > 0.3
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    # Círculo para el keypoint
                    cv2.circle(img, (x, y), 3, color, -1)
                    
                    # Texto con el número del keypoint (solo para COCO)
                    if i < len(COCO_KEYPOINT_NAMES):
                        cv2.putText(img, str(i), (x+5, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Añadir texto del modelo
        cv2.putText(img, f"{model_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

class TestVideoProcessor:
    """
    Procesador de video duplicado para el sistema de testing
    Extrae frames y los prepara para procesamiento
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_frames = []
    
    def extract_frames_from_video(self, video_path: Path, camera_id: int, 
                                 output_dir: Path) -> List[TestPhotoModel]:
        """
        Extraer frames de un video y crear modelos de foto
        
        Args:
            video_path: Ruta al video
            camera_id: ID de la cámara
            output_dir: Directorio donde guardar frames
            
        Returns:
            Lista de modelos de foto creados
        """
        logger.info(f"🎬 Extrayendo frames de {video_path} (cámara {camera_id})")
        
        # Crear directorio para frames de esta cámara
        camera_frames_dir = output_dir / f"camera{camera_id}"
        camera_frames_dir.mkdir(parents=True, exist_ok=True)
        
        photos = []
        
        try:
            # Abrir video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"No se pudo abrir video: {video_path}")
                return photos
            
            # Obtener información del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📹 Video info: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")
            
            # Calcular intervalo de frames para alcanzar target_fps
            frame_interval = max(1, int(fps / self.config.target_fps)) if fps > 0 else 1
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < self.config.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extraer solo cada frame_interval frames
                if frame_count % frame_interval == 0:
                    # Guardar frame
                    frame_filename = f"frame_{extracted_count:04d}.jpg"
                    frame_path = camera_frames_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Crear modelo de foto
                    photo = TestPhotoModel(
                        image_path=str(frame_path),
                        camera_id=camera_id,
                        frame_number=extracted_count
                    )
                    photos.append(photo)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"✅ Extraídos {extracted_count} frames de cámara {camera_id}")
            
        except Exception as e:
            logger.error(f"Error extrayendo frames de {video_path}: {e}")
        
        return photos

class TestMMPoseWrapper:
    """
    Wrapper de MMPose duplicado para el sistema de testing
    Maneja la inicialización y uso de los 4 modelos configurados
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.inferencers = {}
        self.device_assignments = {}
        self.initialized = False
        
        # Rutas a los modelos en el proyecto principal
        self.models_base_path = Path(__file__).parent.parent.parent.parent / 'mmpose_models'
        self.configs_path = self.models_base_path / 'configs' / 'pose2d'
        self.checkpoints_path = self.models_base_path / 'checkpoints'
    
    def initialize_models(self) -> bool:
        """Inicializar todos los modelos MMPose"""
        try:
            # Verificar que MMPose esté disponible
            try:
                from mmpose.apis import MMPoseInferencer
            except ImportError:
                logger.error("❌ MMPose no está instalado. Ejecuta: pip install mmpose")
                return False
            
            logger.info("🤖 Inicializando modelos MMPose para testing...")
            
            # Verificar GPU
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA no disponible, usando CPU")
                primary_gpu = 'cpu'
                secondary_gpu = 'cpu'
            else:
                primary_gpu = self.config.primary_gpu
                secondary_gpu = self.config.secondary_gpu
                logger.info(f"🎮 Usando GPUs: {primary_gpu}, {secondary_gpu}")
            
            success_count = 0
            
            # Inicializar cada modelo
            for model_name, model_config in MMPOSE_MODELS.items():
                try:
                    logger.info(f"🔧 Inicializando {model_name}...")
                    
                    # Construir rutas
                    config_path = self.configs_path / f"{model_config['config']}.py"
                    checkpoint_path = self.checkpoints_path / model_config['checkpoint']
                    
                    # Verificar que los archivos existen
                    if not config_path.exists():
                        logger.error(f"❌ Config no encontrado: {config_path}")
                        logger.info(f"   Ejecuta: python download_models.py")
                        continue
                    
                    if not checkpoint_path.exists():
                        logger.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
                        logger.info(f"   Ejecuta: python download_models.py")
                        continue
                    
                    # Determinar dispositivo
                    device = model_config['gpu']
                    if device == 'cuda:1' and torch.cuda.device_count() <= 1:
                        device = primary_gpu
                    if device.startswith('cuda') and not torch.cuda.is_available():
                        device = 'cpu'
                    
                    # Crear inferencer
                    inferencer = MMPoseInferencer(
                        pose2d=str(config_path),
                        pose2d_weights=str(checkpoint_path),
                        device=device
                    )
                    
                    self.inferencers[model_name] = inferencer
                    self.device_assignments[model_name] = device
                    success_count += 1
                    
                    logger.info(f"✅ {model_name} inicializado en {device}")
                    
                except Exception as e:
                    logger.error(f"❌ Error inicializando {model_name}: {e}")
                    continue
            
            self.initialized = success_count > 0
            
            if self.initialized:
                logger.info(f"🎯 MMPose testing inicializado: {success_count}/{len(MMPOSE_MODELS)} modelos")
            else:
                logger.error("❌ No se pudo inicializar ningún modelo MMPose")
                logger.info("💡 Asegúrate de ejecutar: python download_models.py")
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"❌ Error general inicializando MMPose: {e}")
            return False
    
    def process_photo(self, photo: TestPhotoModel) -> bool:
        """
        Procesar una foto con todos los modelos disponibles
        
        Args:
            photo: Modelo de foto a procesar
            
        Returns:
            True si al menos un modelo procesó correctamente
        """
        if not self.initialized:
            logger.error("MMPose no está inicializado")
            return False
        
        success_count = 0
        
        for model_name, inferencer in self.inferencers.items():
            try:
                start_time = time.time()
                
                # Realizar inferencia
                results = inferencer(str(photo.image_path))
                
                processing_time = time.time() - start_time
                
                # Procesar resultados
                if results and 'predictions' in results and len(results['predictions']) > 0:
                    prediction = results['predictions'][0]  # Primera persona detectada
                    
                    # Extraer keypoints y confianzas
                    keypoints = np.array(prediction.get('keypoints', []))
                    keypoint_scores = np.array(prediction.get('keypoint_scores', []))
                    
                    # Filtrar por confianza
                    valid_mask = keypoint_scores >= self.config.confidence_threshold
                    
                    # Añadir a la foto
                    photo.add_keypoints(
                        model_name=model_name,
                        keypoints=keypoints,
                        confidence=keypoint_scores,
                        processing_time=processing_time
                    )
                    
                    success_count += 1
                    
                    logger.debug(f"✅ {model_name}: {np.sum(valid_mask)}/{len(keypoints)} keypoints válidos")
                    
                else:
                    logger.warning(f"⚠️ {model_name}: No se detectó persona en frame {photo.frame_number}")
                    
                    # Añadir resultado vacío
                    photo.add_keypoints(
                        model_name=model_name,
                        keypoints=np.array([]),
                        confidence=np.array([]),
                        processing_time=processing_time
                    )
                    
            except Exception as e:
                logger.error(f"❌ Error procesando con {model_name}: {e}")
                continue
        
        return success_count > 0
    
    def cleanup(self):
        """Limpiar recursos"""
        try:
            for model_name in list(self.inferencers.keys()):
                del self.inferencers[model_name]
            
            self.inferencers.clear()
            self.device_assignments.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("🧹 Modelos MMPose limpiados")
            
        except Exception as e:
            logger.error(f"Error limpiando modelos: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del wrapper"""
        return {
            'initialized': self.initialized,
            'models_loaded': len(self.inferencers),
            'available_models': list(self.inferencers.keys()),
            'device_assignments': self.device_assignments.copy(),
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

def run_independent_test():
    """
    Función principal para ejecutar el test independiente
    Procesa los videos de las 3 cámaras con los 4 modelos MMPose
    """
    logger.info("🦴 INICIANDO TEST INDEPENDIENTE DE PROCESAMIENTO")
    logger.info("=" * 60)
    
    # Configuración
    config = ProcessingConfig()
    
    # Verificar que existen los videos
    missing_videos = []
    for camera_id, video_path in VIDEO_PATHS.items():
        if not video_path.exists():
            missing_videos.append(f"{camera_id}: {video_path}")
    
    if missing_videos:
        logger.error("❌ Videos faltantes:")
        for missing in missing_videos:
            logger.error(f"   {missing}")
        return False
    
    logger.info("✅ Todos los videos encontrados")
    
    # Crear directorios de salida
    for path in [FRAMES_PATH, KEYPOINTS_PATH, VISUALIZATIONS_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Inicializar procesadores
    logger.info("\n🔧 Inicializando componentes...")
    video_processor = TestVideoProcessor(config)
    mmpose_wrapper = TestMMPoseWrapper(config)
    
    # Inicializar MMPose
    if not mmpose_wrapper.initialize_models():
        logger.error("❌ No se pudo inicializar MMPose")
        logger.info("💡 Ejecuta: python download_models.py")
        return False
    
    try:
        total_photos = []
        
        # Procesar cada cámara
        for camera_id_str, video_path in VIDEO_PATHS.items():
            camera_id = int(camera_id_str.replace('camera', ''))
            
            logger.info(f"\n📹 Procesando {camera_id_str}...")
            
            # Extraer frames
            photos = video_processor.extract_frames_from_video(
                video_path=video_path,
                camera_id=camera_id,
                output_dir=FRAMES_PATH
            )
            
            if not photos:
                logger.warning(f"⚠️ No se extrajeron frames de {camera_id_str}")
                continue
            
            # Procesar frames con MMPose
            logger.info(f"🤖 Procesando {len(photos)} frames con MMPose...")
            
            processed_count = 0
            for i, photo in enumerate(photos):
                if mmpose_wrapper.process_photo(photo):
                    processed_count += 1
                
                # Mostrar progreso cada 10 frames
                if (i + 1) % 10 == 0:
                    logger.info(f"   Progreso: {i+1}/{len(photos)} frames")
            
            logger.info(f"✅ {camera_id_str}: {processed_count}/{len(photos)} frames procesados")
            
            # Guardar keypoints
            camera_keypoints_dir = KEYPOINTS_PATH / f"camera{camera_id}"
            for photo in photos:
                photo.save_keypoints(camera_keypoints_dir)
            
            # Crear visualizaciones
            logger.info(f"🎨 Creando visualizaciones para {camera_id_str}...")
            camera_vis_dir = VISUALIZATIONS_PATH / f"camera{camera_id}"
            for photo in photos:
                photo.create_visualization(camera_vis_dir)
            
            total_photos.extend(photos)
        
        # Resumen final
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMEN DEL TEST INDEPENDIENTE")
        logger.info("=" * 60)
        
        logger.info(f"📹 Videos procesados: {len(VIDEO_PATHS)}")
        logger.info(f"🖼️ Total frames extraídos: {len(total_photos)}")
        
        # Estadísticas por modelo
        model_stats = {}
        for photo in total_photos:
            for model in photo.get_all_models():
                if model not in model_stats:
                    model_stats[model] = {'total': 0, 'with_keypoints': 0}
                
                model_stats[model]['total'] += 1
                keypoints, confidence = photo.get_keypoints(model)
                if len(keypoints) > 0 and np.any(confidence > config.confidence_threshold):
                    model_stats[model]['with_keypoints'] += 1
        
        logger.info("\n🤖 Estadísticas por modelo:")
        for model, stats in model_stats.items():
            detection_rate = (stats['with_keypoints'] / stats['total']) * 100 if stats['total'] > 0 else 0
            logger.info(f"   {model}: {stats['with_keypoints']}/{stats['total']} frames ({detection_rate:.1f}%)")
        
        logger.info(f"\n📁 Resultados guardados en:")
        logger.info(f"   Frames: {FRAMES_PATH}")
        logger.info(f"   Keypoints: {KEYPOINTS_PATH}")
        logger.info(f"   Visualizaciones: {VISUALIZATIONS_PATH}")
        
        logger.info("\n🎉 TEST INDEPENDIENTE COMPLETADO EXITOSAMENTE")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error durante el test: {e}")
        return False
        
    finally:
        # Limpiar recursos
        mmpose_wrapper.cleanup()

if __name__ == "__main__":
    success = run_independent_test()
    sys.exit(0 if success else 1)

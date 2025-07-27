"""
Wrapper simplificado para modelos MMPose
Usa la API simple de MMPoseInferencer con rutas directas
"""
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

# Lista de nombres de keypoints COCO
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class MMPoseInferencerWrapper:
    """
    Wrapper simplificado para manejar múltiples instancias de MMPoseInferencer
    Usa rutas directas a configs y checkpoints
    """
    
    def __init__(self):
        self.inferencers: Dict[str, Any] = {}
        self.device_assignments: Dict[str, str] = {}
        self.model_configs: Dict[str, Dict[str, str]] = {}
        self.initialized = False
        self.models_dir = Path(__file__).parent.parent.parent / "mmpose_models"
    
    
    def initialize_models(self) -> bool:
        """
        Inicializar todos los modelos MMPose configurados
        
        Returns:
            True si todos los modelos se inicializaron correctamente
        """
        try:
            from mmpose.apis import MMPoseInferencer
            
            logger.info("Inicializando modelos MMPose...")
            
            # Definir configuraciones de modelos con rutas directas
            self.model_configs = {
                'hrnet_w48': {
                    'config': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
                    'checkpoint': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
                    'keypoints': 17,
                    'gpu': 'cpu'
                },
                'vitpose_huge': {
                    'config': 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py',
                    'checkpoint': 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth',
                    'keypoints': 17,
                    'gpu': 'cpu'
                }
            }
            
            # Verificar disponibilidad de GPU
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🎮 Dispositivo: {device}")
            
            success_count = 0
            
            # Inicializar cada modelo
            for model_name, config in self.model_configs.items():
                try:
                    config_path = self.models_dir / config['config']
                    checkpoint_path = self.models_dir / config['checkpoint']
                    
                    # Verificar archivos existen
                    if not config_path.exists():
                        logger.error(f"❌ Config no encontrado: {config_path}")
                        continue
                    
                    if not checkpoint_path.exists():
                        logger.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
                        continue
                    
                    logger.info(f"🔧 Inicializando {model_name}...")
                    
                    # Crear inferencer con rutas directas
                    inferencer = MMPoseInferencer(
                        pose2d=str(config_path),
                        pose2d_weights=str(checkpoint_path),
                        device=device
                    )
                    
                    self.inferencers[model_name] = inferencer
                    self.device_assignments[model_name] = device
                    success_count += 1
                    
                    logger.info(f"✅ {model_name} inicializado correctamente")
                    
                except Exception as e:
                    logger.error(f"❌ Error inicializando {model_name}: {e}")
                    continue
            
            self.initialized = success_count > 0
            
            if self.initialized:
                logger.info(f"🎯 MMPose inicializado: {success_count}/{len(self.model_configs)} modelos")
                self._log_model_summary()
            else:
                logger.error("❌ No se pudo inicializar ningún modelo MMPose")
            
            return self.initialized
            
        except ImportError as e:
            logger.error(f"❌ MMPose no está instalado correctamente: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error inicializando MMPose: {e}")
            return False
    
    def _log_model_summary(self):
        """Mostrar resumen de modelos inicializados"""
        logger.info("📊 Resumen de modelos MMPose:")
        for model_name, device in self.device_assignments.items():
            config = self.model_configs[model_name]
            keypoints = config.get('keypoints', 'Unknown')
            logger.info(f"  • {model_name}: {keypoints} keypoints en {device}")
    
    def get_inferencer(self, model_name: str) -> Optional[Any]:
        """
        Obtener inferencer para un modelo específico
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Instancia del inferencer o None si no existe
        """
        return self.inferencers.get(model_name)
    
    def get_all_inferencers(self) -> Dict[str, Any]:
        """Obtener todos los inferencers inicializados"""
        return self.inferencers.copy()
    
    def is_model_available(self, model_name: str) -> bool:
        """Verificar si un modelo está disponible"""
        return model_name in self.inferencers
    
    def get_available_models(self) -> List[str]:
        """Obtener lista de modelos disponibles"""
        return list(self.inferencers.keys())
    
    def inference_single_image(self, model_name: str, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Realizar inferencia en una imagen con un modelo específico
        
        Args:
            model_name: Nombre del modelo a usar
            image_path: Ruta a la imagen
            
        Returns:
            Resultados de la inferencia o None si error
        """
        if not self.is_model_available(model_name):
            logger.error(f"Modelo no disponible: {model_name}")
            return None
        
        try:
            inferencer = self.inferencers[model_name]
            start_time = time.time()
            
            # Realizar inferencia usando la API simple
            result_generator = inferencer(
                image_path,
                return_vis=True,
                vis_out_dir=None,
                pred_out_dir=None,
                show=False
            )
            
            results = next(result_generator)
            processing_time = time.time() - start_time
            
            # Procesar resultados
            if results and 'predictions' in results and len(results['predictions']) > 0:
                prediction = results['predictions'][0]  # Primera persona detectada
                
                return {
                    'model_name': model_name,
                    'image_path': image_path,
                    'keypoints': np.array(prediction.get('keypoints', [])),
                    'keypoint_scores': np.array(prediction.get('keypoint_scores', [])),
                    'bbox': prediction.get('bbox', None),
                    'bbox_score': prediction.get('bbox_score', 0.0),
                    'processing_time': processing_time,
                    'device': self.device_assignments[model_name]
                }
            else:
                logger.warning(f"No se detectaron personas en {image_path} con {model_name}")
                return {
                    'model_name': model_name,
                    'image_path': image_path,
                    'keypoints': np.array([]),
                    'keypoint_scores': np.array([]),
                    'bbox': None,
                    'bbox_score': 0.0,
                    'processing_time': processing_time,
                    'device': self.device_assignments[model_name]
                }
                
        except Exception as e:
            logger.error(f"Error en inferencia {model_name} para {image_path}: {e}")
            return None
    
    def inference_batch(self, model_name: str, image_paths: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Realizar inferencia en lote para múltiples imágenes
        
        Args:
            model_name: Nombre del modelo
            image_paths: Lista de rutas a imágenes
            
        Returns:
            Lista de resultados (puede contener None para errores)
        """
        results = []
        
        if not self.is_model_available(model_name):
            logger.error(f"Modelo no disponible: {model_name}")
            return [None] * len(image_paths)
        
        logger.info(f"🔄 Procesando lote de {len(image_paths)} imágenes con {model_name}")
        
        for image_path in image_paths:
            result = self.inference_single_image(model_name, image_path)
            results.append(result)
        
        success_count = sum(1 for r in results if r is not None)
        logger.info(f"✅ Lote completado: {success_count}/{len(image_paths)} exitosos")
        
        return results
    
    def process_video(self, model_name: str, video_path: str, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Procesar un video completo frame por frame
        
        Args:
            model_name: Nombre del modelo a usar
            video_path: Ruta al video
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Lista de resultados por frame
        """
        if not self.is_model_available(model_name):
            logger.error(f"Modelo no disponible: {model_name}")
            return []
        
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                logger.error(f"Video no encontrado: {video_path}")
                return []
            
            logger.info(f"🎬 Procesando video: {video_path.name} con {model_name}")
            
            # Usar la API simple de MMPose para procesar video
            inferencer = self.inferencers[model_name]
            start_time = time.time()
            
            result_generator = inferencer(
                str(video_path),
                return_vis=True,
                vis_out_dir=output_dir,
                pred_out_dir=output_dir,
                show=False
            )
            
            results = []
            frame_count = 0
            
            for result in result_generator:
                frame_count += 1
                if result and 'predictions' in result:
                    results.append({
                        'frame': frame_count,
                        'predictions': result['predictions'],
                        'model_name': model_name
                    })
                
                if frame_count % 30 == 0:  # Log cada 30 frames
                    logger.info(f"Procesados {frame_count} frames...")
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Video procesado: {frame_count} frames en {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error procesando video {video_path}: {e}")
            return []
    
    def get_model_keypoint_count(self, model_name: str) -> int:
        """Obtener número de keypoints de un modelo"""
        if model_name in self.model_configs:
            return self.model_configs[model_name].get('keypoints', 0)
        return 0
    
    def get_model_device(self, model_name: str) -> str:
        """Obtener dispositivo asignado a un modelo"""
        return self.device_assignments.get(model_name, 'unknown')
    
    def cleanup(self):
        """Limpiar recursos de todos los modelos"""
        try:
            for model_name in list(self.inferencers.keys()):
                del self.inferencers[model_name]
            
            self.inferencers.clear()
            self.device_assignments.clear()
            
            # Limpiar cache de GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("🧹 Modelos MMPose limpiados")
            
        except Exception as e:
            logger.error(f"Error limpiando modelos MMPose: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual del wrapper"""
        return {
            'initialized': self.initialized,
            'models_loaded': len(self.inferencers),
            'available_models': self.get_available_models(),
            'device_assignments': self.device_assignments.copy(),
            'cuda_available': torch.cuda.is_available(),
            'models_dir': str(self.models_dir)
        }

# Instancia global del wrapper
mmpose_wrapper = MMPoseInferencerWrapper()

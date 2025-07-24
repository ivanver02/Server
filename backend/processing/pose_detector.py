"""
Wrapper para modelos MMPose
Maneja la inicialización y uso de múltiples modelos de detección de pose
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class MMPoseInferencerWrapper:
    """
    Wrapper para manejar múltiples instancias de MMPoseInferencer
    Optimizado para procesamiento en múltiples GPUs
    """
    
    def __init__(self):
        self.inferencers: Dict[str, Any] = {}
        self.device_assignments: Dict[str, str] = {}
        self.model_configs: Dict[str, Dict[str, str]] = {}
        self.initialized = False
    
    def initialize_models(self) -> bool:
        """
        Inicializar todos los modelos MMPose configurados
        
        Returns:
            True si todos los modelos se inicializaron correctamente
        """
        try:
            from mmpose.apis import MMPoseInferencer
            from config import mmpose_config, processing_config
            
            logger.info("🤖 Inicializando modelos MMPose...")
            
            # Verificar disponibilidad de GPU
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA no disponible, usando CPU")
                device_primary = 'cpu'
                device_secondary = 'cpu'
            else:
                device_primary = processing_config.primary_gpu
                device_secondary = processing_config.secondary_gpu
                logger.info(f"🎮 GPUs disponibles: {device_primary}, {device_secondary}")
            
            self.model_configs = mmpose_config.model_configs
            success_count = 0
            
            # Inicializar cada modelo
            for model_name, config in self.model_configs.items():
                try:
                    config_path = mmpose_config.models_dir / config['config']
                    checkpoint_path = mmpose_config.models_dir / config['checkpoint']
                    
                    # Verificar archivos existen
                    if not config_path.exists():
                        logger.error(f"❌ Config no encontrado: {config_path}")
                        continue
                    
                    if not checkpoint_path.exists():
                        logger.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
                        continue
                    
                    # Asignar dispositivo según configuración
                    device = config.get('gpu', device_primary)
                    if device not in ['cpu', device_primary, device_secondary]:
                        device = device_primary
                    
                    logger.info(f"🔧 Inicializando {model_name} en {device}...")
                    
                    # Crear inferencer
                    inferencer = MMPoseInferencer(
                        pose2d=str(config_path),
                        pose2d_weights=str(checkpoint_path),
                        device=device
                    )
                    
                    self.inferencers[model_name] = inferencer
                    self.device_assignments[model_name] = device
                    success_count += 1
                    
                    logger.info(f"✅ {model_name} inicializado correctamente en {device}")
                    
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
            
            # Realizar inferencia
            results = inferencer(image_path)
            
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
            'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'gpu_memory_cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }

# Instancia global del wrapper
mmpose_wrapper = MMPoseInferencerWrapper()

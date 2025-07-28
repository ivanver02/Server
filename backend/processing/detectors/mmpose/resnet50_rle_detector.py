"""
Detector ResNet-50 RLE - MMPose
Implementación específica para el modelo ResNet-50 con RLE
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from ..base import BasePoseDetector

logger = logging.getLogger(__name__)


class ResNet50RLEDetector(BasePoseDetector):
    """
    Detector específico para ResNet-50 RLE de MMPose
    Modelo ligero y rápido con buena precisión
    """
    
    def __init__(self):
        super().__init__("resnet50_rle")
        self.inferencer = None
        self.device = 'cuda' if self._is_cuda_available() else 'cpu'
    
    def _is_cuda_available(self) -> bool:
        """Verificar si CUDA está disponible"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        """Inicializar el detector ResNet-50 RLE"""
        try:
            # Importar MMPose
            try:
                from mmpose.apis import MMPoseInferencer
            except ImportError as e:
                logger.error(f"MMPose no está instalado: {e}")
                return False
            
            # Usar configuración centralizada
            from config import mmpose_config
            config = mmpose_config.resnet50_rle
            
            pose2d_config = config['config']
            pose2d_weights = config['checkpoint']
            
            # Crear inferencer
            self.inferencer = MMPoseInferencer(
                pose2d=pose2d_config,
                pose2d_weights=pose2d_weights,
                device=self.device
            )
            
            self.is_initialized = True
            logger.info(f"ResNet50RLEDetector inicializado en {self.device} con config: {pose2d_config}")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando ResNet50RLEDetector: {e}")
            return False
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detectar keypoints en un frame usando ResNet-50 RLE"""
        if not self.is_initialized:
            logger.warning("ResNet50RLEDetector no inicializado")
            return None, None
        
        try:
            # Ejecutar inferencia
            results = self.inferencer(frame, show=False, return_vis=False)
            
            # Extraer resultados
            if results and 'predictions' in results and len(results['predictions']) > 0:
                prediction = results['predictions'][0]  # Primer detección
                
                # Extraer keypoints y scores
                if 'keypoints' in prediction:
                    keypoints = np.array(prediction['keypoints'])  # Shape: (17, 2)
                    scores = np.array(prediction.get('keypoint_scores', [1.0] * len(keypoints)))  # Shape: (17,)
                    
                    return keypoints, scores
                    
            return None, None
            
        except Exception as e:
            logger.error(f"Error en detección ResNet-50 RLE: {e}")
            return None, None
    
    def get_model_info(self) -> dict:
        """Obtener información del modelo"""
        return {
            'name': 'ResNet-50 RLE',
            'keypoints': 17,
            'type': 'COCO 2D',
            'device': self.device,
            'framework': 'MMPose'
        }
    
    def cleanup(self):
        """Limpiar recursos del detector"""
        if self.inferencer:
            del self.inferencer
            self.inferencer = None
        
        self.is_initialized = False
        logger.info("ResNet50RLEDetector limpiado")

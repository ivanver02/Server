"""
Detector HRNet-W48 - MMPose
Implementación específica para el modelo HRNet-W48
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from ..base import BasePoseDetector

logger = logging.getLogger(__name__)


class HRNetW48Detector(BasePoseDetector):
    """
    Detector específico para HRNet-W48 de MMPose
    Excelente para detección de keypoints con alta resolución
    """
    
    def __init__(self, config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
        super().__init__("hrnet_w48_coco_256x192")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
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
        """Inicializar el detector HRNet-W48"""
        try:
            # Importar MMPose
            try:
                from mmpose.apis import MMPoseInferencer
            except ImportError as e:
                logger.error(f"MMPose no está instalado: {e}")
                return False
            
            # Usar configuración por defecto si no se proporciona
            if not self.config_path or not self.checkpoint_path:
                # Configuración por defecto para HRNet-W48
                pose2d_config = "hrnet-w48_coco_256x192"
                pose2d_weights = None  # MMPose descargará automáticamente
            else:
                pose2d_config = self.config_path
                pose2d_weights = self.checkpoint_path
            
            # Crear inferencer
            self.inferencer = MMPoseInferencer(
                pose2d=pose2d_config,
                pose2d_weights=pose2d_weights,
                device=self.device
            )
            
            self.is_initialized = True
            logger.info(f"HRNetW48Detector inicializado en {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando HRNetW48Detector: {e}")
            return False
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detectar keypoints en un frame usando HRNet-W48"""
        if not self.is_initialized:
            logger.warning("HRNetW48Detector no inicializado")
            return None, None
        
        try:
            # Ejecutar inferencia
            results = self.inferencer(frame, show=False, return_vis=False)
            
            # Extraer resultados
            if results and 'predictions' in results and len(results['predictions']) > 0:
                prediction = results['predictions'][0]  # Primer detección
                
                # Extraer keypoints y scores
                if 'keypoints' in prediction:
                    keypoints = np.array(prediction['keypoints'])  # Shape: (N, 2) o (N, 3)
                    
                    # Extraer confidence scores
                    if keypoints.shape[1] >= 3:
                        scores = keypoints[:, 2]  # Confidence en tercera columna
                        keypoints_2d = keypoints[:, :2]
                    else:
                        scores = np.ones(len(keypoints))  # Scores por defecto
                        keypoints_2d = keypoints
                    
                    return keypoints_2d, scores
            
            # No se detectaron personas
            return None, None
            
        except Exception as e:
            logger.error(f"Error en inferencia HRNet-W48: {e}")
            return None, None

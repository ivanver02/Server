"""
Detector VitPose - MMPose
Implementación específica para el modelo VitPose
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from ..base import BasePoseDetector

logger = logging.getLogger(__name__)


class VitPoseDetector(BasePoseDetector):
    """
    Detector específico para VitPose de MMPose
    Excelente precisión en keypoints de cuerpo completo
    """
    
    def __init__(self, config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
        super().__init__("vitpose_huge_coco")
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
        """Inicializar el detector VitPose"""
        try:
            # Importar MMPose
            try:
                from mmpose.apis import MMPoseInferencer
            except ImportError as e:
                logger.error(f"MMPose no está instalado: {e}")
                return False
            
            # Usar configuración por defecto si no se proporciona
            if not self.config_path or not self.checkpoint_path:
                # Configuración por defecto para VitPose
                pose2d_config = "vitpose-h-multi-coco"
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
            logger.info(f"VitPoseDetector inicializado en {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando VitPoseDetector: {e}")
            return False
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detectar keypoints en un frame usando VitPose"""
        if not self.is_initialized:
            logger.warning("VitPoseDetector no inicializado")
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
            logger.error(f"Error en inferencia VitPose: {e}")
            return None, None

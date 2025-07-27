"""
Detector MMPose para procesamiento de pose escalable
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import time

from ..base import BasePoseDetector
from ...data import KeypointResult

logger = logging.getLogger(__name__)

# Lista de nombres de keypoints COCO (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class MMPoseDetector(BasePoseDetector):
    """
    Detector MMPose que usa archivos locales (.py y .pth)
    Soporte para procesamiento directo de video y frames individuales
    """
    
    def __init__(self, model_name: str, model_path: Optional[Path] = None):
        super().__init__(model_name, model_path)
        self.inferencer = None
        self.config_path = None
        self.checkpoint_path = None
        self.device = 'cuda' if self._is_cuda_available() else 'cpu'
        
    def _is_cuda_available(self) -> bool:
        """Verificar si CUDA está disponible"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _find_model_files(self) -> bool: # Se usará simplemente para ver que existen los archivos necesarios
        """
        Buscar archivos de configuración (.py) y checkpoint (.pth) del modelo
        """
        try:
            if not self.model_path or not self.model_path.exists():
                logger.error(f"Ruta del modelo no válida: {self.model_path}")
                return False
            
            # Buscar archivo de configuración (.py)
            config_files = list(self.model_path.glob("*.py"))
            if not config_files:
                logger.error(f"No se encontró archivo de configuración .py en {self.model_path}")
                return False
            self.config_path = config_files[0]
            
            # Buscar archivo de checkpoint (.pth)
            checkpoint_files = list(self.model_path.glob("*.pth"))
            if not checkpoint_files:
                logger.error(f"No se encontró archivo de checkpoint .pth en {self.model_path}")
                return False
            self.checkpoint_path = checkpoint_files[0]
            
            logger.info(f"Modelo {self.model_name} - Config: {self.config_path.name}, "
                       f"Checkpoint: {self.checkpoint_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error buscando archivos del modelo {self.model_name}: {e}")
            return False
    
    def initialize(self) -> bool:
        """Inicializar el inferencer MMPose"""
        try:
            if not self._find_model_files():
                return False
            
            # Importar MMPose
            try:
                from mmpose.apis import MMPoseInferencer
            except ImportError as e:
                logger.error(f"MMPose no está instalado: {e}")
                return False
            
            # Crear inferencer con archivos locales
            self.inferencer = MMPoseInferencer(
                pose2d=str(self.config_path),
                pose2d_weights=str(self.checkpoint_path),
                device=self.device
            )
            
            self.is_initialized = True
            logger.info(f"MMPoseDetector inicializado: {self.model_name} en {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando MMPoseDetector {self.model_name}: {e}")
            return False
    
    def detect_frame(self, frame: np.ndarray) -> KeypointResult:
        """
        Detectar keypoints en un frame individual
        """
        if not self.is_initialized:
            return KeypointResult(
                success=False,
                error_message="Detector no inicializado"
            )
        
        start_time = time.time()
        
        try:
            # MMPose espera imágenes en formato BGR (OpenCV)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Asegurar formato BGR
                input_frame = frame
            else:
                return KeypointResult(
                    success=False,
                    error_message="Frame debe ser imagen en color (H,W,3)"
                )
            
            # Procesar con MMPose
            results = self.inferencer(input_frame, return_vis=False)
            
            # Extraer resultados
            if results and 'predictions' in results[0]:
                predictions = results[0]['predictions']
                
                if predictions and len(predictions) > 0:
                    # Tomar la primera persona detectada
                    pred = predictions[0]
                    
                    keypoints = np.array(pred['keypoints'])  # (N, 2)
                    scores = np.array(pred['keypoint_scores'])  # (N,)
                    
                    # Extraer bbox si está disponible
                    bbox = None
                    if 'bbox' in pred:
                        bbox = np.array(pred['bbox'])  # [x1, y1, x2, y2]
                    
                    return KeypointResult(
                        success=True,
                        keypoints=keypoints,
                        scores=scores,
                        bbox=bbox,
                        processing_time=time.time() - start_time,
                        metadata={
                            'model_name': self.model_name,
                            'device': self.device,
                            'num_persons': len(predictions)
                        }
                    )
                else:
                    return KeypointResult(
                        success=True,
                        keypoints=np.array([]),
                        scores=np.array([]),
                        processing_time=time.time() - start_time,
                        metadata={'model_name': self.model_name, 'num_persons': 0}
                    )
            else:
                return KeypointResult(
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message="No se obtuvieron predicciones"
                )
                
        except Exception as e:
            return KeypointResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=f"Error en detección: {str(e)}"
            )
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[KeypointResult]:
        """
        Detectar keypoints en un batch de frames
        """
        results = []
        
        for frame in frames:
            result = self.detect_frame(frame)
            results.append(result)
        
        return results
    
    def process_video_synchronized(self, video_path: Path, 
                                 timestamps: List[float]) -> List[KeypointResult]:
        """
        Procesar video extrayendo frames en timestamps específicos
        Para mantener sincronización entre múltiples cámaras
        """
        if not self.is_initialized:
            return [KeypointResult(success=False, error_message="Detector no inicializado") 
                   for _ in timestamps]
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return [KeypointResult(success=False, error_message="No se puede abrir video") 
                       for _ in timestamps]
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            results = []
            
            for timestamp in timestamps:
                # Convertir timestamp a número de frame
                frame_number = int(timestamp * fps)
                
                # Posicionar video en el frame correcto
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    result = self.detect_frame(frame)
                    result.metadata = result.metadata or {}
                    result.metadata.update({
                        'timestamp': timestamp,
                        'frame_number': frame_number,
                        'video_path': str(video_path)
                    })
                    results.append(result)
                else:
                    results.append(KeypointResult(
                        success=False,
                        error_message=f"No se pudo leer frame en timestamp {timestamp}"
                    ))
            
            cap.release()
            return results
            
        except Exception as e:
            return [KeypointResult(success=False, error_message=f"Error procesando video: {e}") 
                   for _ in timestamps]
    
    def cleanup(self):
        """Limpiar recursos del detector"""
        try:
            if self.inferencer:
                # MMPose no requiere limpieza explícita normalmente
                self.inferencer = None
            
            self.is_initialized = False
            logger.info(f"MMPoseDetector {self.model_name} limpiado")
            
        except Exception as e:
            logger.error(f"Error limpiando MMPoseDetector {self.model_name}: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path) if self.model_path else None,
            'config_path': str(self.config_path) if self.config_path else None,
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'device': self.device,
            'is_initialized': self.is_initialized,
            'num_keypoints': self.get_num_keypoints(),
            'keypoint_names': self.get_keypoint_names()
        }
    
    def get_keypoint_names(self) -> List[str]:
        """Obtener nombres de keypoints COCO"""
        return COCO_KEYPOINT_NAMES.copy()
    
    def get_num_keypoints(self) -> int:
        """Obtener número de keypoints (17 para COCO)"""
        return len(COCO_KEYPOINT_NAMES)

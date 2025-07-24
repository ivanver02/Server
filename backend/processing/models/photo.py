"""
Clase Photo para representar frames individuales extraídos de video
Maneja el procesamiento de una imagen individual con un modelo MMPose específico
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Photo:
    """
    Representa un frame individual para procesamiento con MMPose
    
    Attributes:
        patient_id: ID del paciente
        session_id: ID de la sesión
        camera_id: ID de la cámara
        frame_number: Número del frame dentro del chunk
        chunk_number: Número del chunk de video
        model_name: Nombre del modelo MMPose a usar
        image_path: Ruta a la imagen extraída
        keypoints_2d: Coordenadas 2D de keypoints (None hasta procesar)
        confidence: Scores de confianza (None hasta procesar)
    """
    patient_id: str
    session_id: str
    camera_id: int
    frame_number: int
    chunk_number: int
    model_name: str
    image_path: Optional[Path] = None
    keypoints_2d: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        """Validar datos después de inicialización"""
        if self.camera_id < 0:
            raise ValueError("camera_id must be non-negative")
        if self.frame_number < 0:
            raise ValueError("frame_number must be non-negative")
        if self.chunk_number < 0:
            raise ValueError("chunk_number must be non-negative")
    
    @property
    def unique_id(self) -> str:
        """Generar ID único para esta foto"""
        return f"p{self.patient_id}_s{self.session_id}_c{self.camera_id}_ch{self.chunk_number}_f{self.frame_number}"
    
    @property
    def expected_image_path(self) -> Path:
        """Generar ruta esperada para la imagen"""
        from config import data_config
        
        base_dir = (data_config.photos_dir / 
                   f"patient{self.patient_id}" / 
                   f"session{self.session_id}" / 
                   f"camera{self.camera_id}")
        
        # Nombrar archivo con número global (chunk_number * frames_per_chunk + frame_number)
        # Asumiendo ~15 fps y chunks de 5s = ~75 frames por chunk
        global_frame = self.chunk_number * 75 + self.frame_number
        filename = f"{global_frame}.jpg"
        
        return base_dir / filename
    
    def load_image(self) -> Optional[np.ndarray]:
        """
        Cargar imagen desde archivo
        
        Returns:
            Imagen en formato BGR (OpenCV) o None si error
        """
        if not self.image_path or not self.image_path.exists():
            logger.error(f"Imagen no encontrada: {self.image_path}")
            return None
        
        try:
            image = cv2.imread(str(self.image_path))
            if image is None:
                logger.error(f"Error cargando imagen: {self.image_path}")
                return None
            
            logger.debug(f"Imagen cargada: {self.image_path}, Shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error leyendo imagen {self.image_path}: {e}")
            return None
    
    def save_image(self, image: np.ndarray) -> bool:
        """
        Guardar imagen en el path esperado
        
        Args:
            image: Imagen en formato BGR (OpenCV)
            
        Returns:
            True si se guardó correctamente
        """
        try:
            save_path = self.expected_image_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            success = cv2.imwrite(str(save_path), image)
            if success:
                self.image_path = save_path
                logger.debug(f"Imagen guardada: {save_path}")
                return True
            else:
                logger.error(f"Error guardando imagen: {save_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error salvando imagen: {e}")
            return False
    
    def process_with_mmpose(self, inferencer) -> bool:
        """
        Procesar imagen con modelo MMPose
        
        Args:
            inferencer: Instancia de MMPoseInferencer configurada
            
        Returns:
            True si el procesamiento fue exitoso
        """
        if not self.image_path or not self.image_path.exists():
            logger.error(f"No hay imagen para procesar: {self.unique_id}")
            return False
        
        try:
            import time
            start_time = time.time()
            
            # Procesar con MMPose
            results = inferencer(str(self.image_path))
            
            # Extraer resultados
            if results and len(results['predictions']) > 0:
                prediction = results['predictions'][0]  # Primera persona detectada
                
                # Obtener keypoints y confianza
                if 'keypoints' in prediction:
                    keypoints = np.array(prediction['keypoints'])  # Shape: (N, 2)
                    self.keypoints_2d = keypoints
                    
                if 'keypoint_scores' in prediction:
                    confidence = np.array(prediction['keypoint_scores'])  # Shape: (N,)
                    self.confidence = confidence
                
                self.processing_time = time.time() - start_time
                
                logger.debug(f"Procesamiento exitoso {self.unique_id}: "
                           f"{len(self.keypoints_2d) if self.keypoints_2d is not None else 0} keypoints, "
                           f"tiempo: {self.processing_time:.3f}s")
                
                return True
            else:
                logger.warning(f"No se detectaron personas en {self.unique_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error procesando {self.unique_id} con MMPose: {e}")
            return False
    
    def save_keypoints(self) -> bool:
        """
        Guardar keypoints 2D y confianza en archivos .npy
        
        Returns:
            True si se guardaron correctamente
        """
        if self.keypoints_2d is None or self.confidence is None:
            logger.error(f"No hay keypoints para guardar: {self.unique_id}")
            return False
        
        try:
            from config import data_config
            
            # Directorios base
            base_dir = (data_config.unprocessed_dir / 
                       f"patient{self.patient_id}" / 
                       f"session{self.session_id}" / 
                       f"camera{self.camera_id}" / 
                       "2D")
            
            points_dir = base_dir / "points" / self.model_name
            confidence_dir = base_dir / "confidence" / self.model_name
            
            points_dir.mkdir(parents=True, exist_ok=True)
            confidence_dir.mkdir(parents=True, exist_ok=True)
            
            # Nombres de archivo (número global)
            global_frame = self.chunk_number * 75 + self.frame_number
            
            # Guardar keypoints
            points_file = points_dir / f"{global_frame}.npy"
            np.save(points_file, self.keypoints_2d)
            
            # Guardar confianza
            confidence_file = confidence_dir / f"{global_frame}.npy"
            np.save(confidence_file, self.confidence)
            
            logger.debug(f"Keypoints guardados: {points_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando keypoints {self.unique_id}: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen del estado de procesamiento"""
        return {
            'unique_id': self.unique_id,
            'patient_id': self.patient_id,
            'session_id': self.session_id,
            'camera_id': self.camera_id,
            'frame_number': self.frame_number,
            'chunk_number': self.chunk_number,
            'model_name': self.model_name,
            'image_exists': self.image_path is not None and self.image_path.exists(),
            'keypoints_detected': self.keypoints_2d is not None,
            'keypoints_count': len(self.keypoints_2d) if self.keypoints_2d is not None else 0,
            'processing_time': self.processing_time,
            'confidence_mean': float(np.mean(self.confidence)) if self.confidence is not None else None
        }

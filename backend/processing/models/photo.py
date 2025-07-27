"""
Clase Photo para procesamiento de imágenes individuales con MMPose
Maneja la extracción de keypoints de una imagen específica
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class KeypointResult:
    """Resultado de la detección de keypoints"""
    success: bool
    keypoints: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None

class Photo:
    """
    Maneja el procesamiento de una imagen individual con MMPose
    
    Atributos:
        patient_id: ID del paciente
        session_id: ID de la sesión
        camera_id: ID de la cámara
        frame_number: Número del frame
        chunk_number: Número del chunk
        model_name: Nombre del modelo MMPose a usar
        image_path: Ruta al archivo de imagen (se asigna después de guardar)
    """
    
    def __init__(self, patient_id: str, session_id: str, camera_id: int,
                 frame_number: int, chunk_number: int, model_name: str,
                 image_path: Optional[Path] = None):
        self.patient_id = patient_id
        self.session_id = session_id
        self.camera_id = camera_id
        self.frame_number = frame_number
        self.chunk_number = chunk_number
        self.model_name = model_name
        self.image_path = Path(image_path) if image_path else None
        
        # Estado del procesamiento
        self._keypoint_result: Optional[KeypointResult] = None
        self._processed = False
        
        # Validaciones
        if self.camera_id < 0:
            raise ValueError("camera_id must be non-negative")
        if self.frame_number < 0:
            raise ValueError("frame_number must be non-negative")
        if self.chunk_number < 0:
            raise ValueError("chunk_number must be non-negative")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
    
    @property
    def unique_id(self) -> str:
        """ID único para esta foto"""
        return f"p{self.patient_id}_s{self.session_id}_c{self.camera_id}_f{self.frame_number}_ch{self.chunk_number}_{self.model_name}"
    
    @property
    def expected_image_path(self) -> Path:
        """Ruta esperada para la imagen"""
        try:
            from config import data_config
            
            base_dir = (data_config.photos_dir / 
                       f"patient{self.patient_id}" / 
                       f"session{self.session_id}" / 
                       f"camera{self.camera_id}")
            
            base_dir.mkdir(parents=True, exist_ok=True)
            
            return base_dir / f"frame_{self.frame_number}_chunk_{self.chunk_number}.jpg"
            
        except Exception as e:
            logger.error(f"Error calculando ruta de imagen: {e}")
            # Fallback a ruta temporal
            return Path(f"temp_frame_{self.frame_number}_{self.chunk_number}.jpg")
    
    @property
    def expected_keypoints_path(self) -> Path:
        """Ruta esperada para los keypoints"""
        try:
            from config import data_config
            
            base_dir = (data_config.keypoints_2d_dir / 
                       f"patient{self.patient_id}" / 
                       f"session{self.session_id}" / 
                       f"camera{self.camera_id}" /
                       self.model_name)
            
            base_dir.mkdir(parents=True, exist_ok=True)
            
            return base_dir / f"frame_{self.frame_number}_chunk_{self.chunk_number}.json"
            
        except Exception as e:
            logger.error(f"Error calculando ruta de keypoints: {e}")
            # Fallback a ruta temporal
            return Path(f"temp_keypoints_{self.frame_number}_{self.chunk_number}_{self.model_name}.json")
    
    def save_image(self, image: np.ndarray) -> bool:
        """
        Guardar imagen en disco
        
        Args:
            image: Array numpy con la imagen
            
        Returns:
            True si se guardó correctamente
        """
        try:
            if image is None or image.size == 0:
                logger.error(f"Imagen vacía para {self.unique_id}")
                return False
            
            image_path = self.expected_image_path
            
            # Crear directorio si no existe
            image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar imagen
            success = cv2.imwrite(str(image_path), image)
            
            if success:
                self.image_path = image_path
                logger.debug(f"Imagen guardada: {image_path}")
                return True
            else:
                logger.error(f"Error guardando imagen en {image_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error guardando imagen para {self.unique_id}: {e}")
            return False
    
    def load_image(self) -> Optional[np.ndarray]:
        """
        Cargar imagen desde disco
        
        Returns:
            Array numpy con la imagen o None si hay error
        """
        try:
            if not self.image_path or not self.image_path.exists():
                logger.error(f"Imagen no encontrada: {self.image_path}")
                return None
            
            image = cv2.imread(str(self.image_path))
            
            if image is None:
                logger.error(f"Error cargando imagen: {self.image_path}")
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error cargando imagen para {self.unique_id}: {e}")
            return None
    
    def process_with_mmpose(self, inferencer: Any) -> bool:
        """
        Procesar imagen con modelo MMPose
        
        Args:
            inferencer: Objeto inferencer de MMPose
            
        Returns:
            True si el procesamiento fue exitoso
        """
        try:
            if not self.image_path or not self.image_path.exists():
                logger.error(f"Imagen no disponible para procesar: {self.unique_id}")
                return False
            
            start_time = time.time()
            
            # Cargar imagen
            image = self.load_image()
            if image is None:
                return False
            
            logger.debug(f"Procesando {self.unique_id} con modelo {self.model_name}")
            
            # Procesar con MMPose
            result = inferencer(image)
            
            processing_time = time.time() - start_time
            
            # Extraer información del resultado
            if result and len(result) > 0:
                # MMPose suele devolver una lista de detecciones
                detection = result[0] if isinstance(result, list) else result
                
                # Extraer keypoints y scores
                keypoints = None
                scores = None
                bbox = None
                
                if hasattr(detection, 'pred_instances'):
                    pred_instances = detection.pred_instances
                    
                    if hasattr(pred_instances, 'keypoints'):
                        keypoints = pred_instances.keypoints.cpu().numpy()
                    
                    if hasattr(pred_instances, 'keypoint_scores'):
                        scores = pred_instances.keypoint_scores.cpu().numpy()
                    
                    if hasattr(pred_instances, 'bboxes'):
                        bbox = pred_instances.bboxes.cpu().numpy()
                
                # Crear resultado
                self._keypoint_result = KeypointResult(
                    success=True,
                    keypoints=keypoints,
                    scores=scores,
                    bbox=bbox,
                    processing_time=processing_time
                )
                
                self._processed = True
                
                logger.debug(f"Procesamiento exitoso para {self.unique_id}: "
                           f"{processing_time:.3f}s, "
                           f"keypoints shape: {keypoints.shape if keypoints is not None else 'None'}")
                
                return True
            
            else:
                self._keypoint_result = KeypointResult(
                    success=False,
                    processing_time=processing_time,
                    error_message="No se detectaron personas en la imagen"
                )
                
                logger.warning(f"No se detectaron personas en {self.unique_id}")
                return False
                
        except Exception as e:
            error_msg = f"Error procesando {self.unique_id}: {e}"
            logger.error(error_msg)
            
            self._keypoint_result = KeypointResult(
                success=False,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0,
                error_message=str(e)
            )
            
            return False
    
    def save_keypoints(self) -> bool:
        """
        Guardar keypoints en formato JSON
        
        Returns:
            True si se guardaron correctamente
        """
        try:
            if not self._keypoint_result or not self._keypoint_result.success:
                logger.error(f"No hay keypoints válidos para guardar: {self.unique_id}")
                return False
            
            keypoints_path = self.expected_keypoints_path
            
            # Crear directorio si no existe
            keypoints_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Preparar datos para JSON
            data = {
                'photo_info': {
                    'patient_id': self.patient_id,
                    'session_id': self.session_id,
                    'camera_id': self.camera_id,
                    'frame_number': self.frame_number,
                    'chunk_number': self.chunk_number,
                    'model_name': self.model_name,
                    'unique_id': self.unique_id
                },
                'processing_info': {
                    'processing_time': self._keypoint_result.processing_time,
                    'timestamp': time.time()
                },
                'keypoints_data': {
                    'keypoints': self._keypoint_result.keypoints.tolist() if self._keypoint_result.keypoints is not None else None,
                    'scores': self._keypoint_result.scores.tolist() if self._keypoint_result.scores is not None else None,
                    'bbox': self._keypoint_result.bbox.tolist() if self._keypoint_result.bbox is not None else None
                }
            }
            
            # Guardar JSON
            with open(keypoints_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Keypoints guardados: {keypoints_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando keypoints para {self.unique_id}: {e}")
            return False
    
    def load_keypoints(self) -> Optional[KeypointResult]:
        """
        Cargar keypoints desde archivo JSON
        
        Returns:
            KeypointResult con los datos cargados o None si hay error
        """
        try:
            keypoints_path = self.expected_keypoints_path
            
            if not keypoints_path.exists():
                logger.debug(f"Archivo de keypoints no encontrado: {keypoints_path}")
                return None
            
            with open(keypoints_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            keypoints_data = data.get('keypoints_data', {})
            
            result = KeypointResult(
                success=True,
                keypoints=np.array(keypoints_data['keypoints']) if keypoints_data.get('keypoints') else None,
                scores=np.array(keypoints_data['scores']) if keypoints_data.get('scores') else None,
                bbox=np.array(keypoints_data['bbox']) if keypoints_data.get('bbox') else None,
                processing_time=data.get('processing_info', {}).get('processing_time', 0.0)
            )
            
            self._keypoint_result = result
            self._processed = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error cargando keypoints para {self.unique_id}: {e}")
            return None
    
    def get_keypoints(self) -> Optional[KeypointResult]:
        """
        Obtener keypoints (carga desde disco si no están en memoria)
        
        Returns:
            KeypointResult con los keypoints o None
        """
        if self._keypoint_result and self._keypoint_result.success:
            return self._keypoint_result
        
        # Intentar cargar desde disco
        return self.load_keypoints()
    
    def is_processed(self) -> bool:
        """
        Verificar si la foto ha sido procesada
        
        Returns:
            True si fue procesada exitosamente
        """
        return self._processed and self._keypoint_result and self._keypoint_result.success
    
    def cleanup_files(self):
        """Limpiar archivos asociados (imagen y keypoints)"""
        try:
            # Eliminar imagen
            if self.image_path and self.image_path.exists():
                self.image_path.unlink()
                logger.debug(f"Imagen eliminada: {self.image_path}")
            
            # Eliminar keypoints
            keypoints_path = self.expected_keypoints_path
            if keypoints_path.exists():
                keypoints_path.unlink()
                logger.debug(f"Keypoints eliminados: {keypoints_path}")
                
        except Exception as e:
            logger.error(f"Error limpiando archivos para {self.unique_id}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de la foto"""
        summary = {
            'unique_id': self.unique_id,
            'patient_id': self.patient_id,
            'session_id': self.session_id,
            'camera_id': self.camera_id,
            'frame_number': self.frame_number,
            'chunk_number': self.chunk_number,
            'model_name': self.model_name,
            'image_path': str(self.image_path) if self.image_path else None,
            'image_exists': self.image_path.exists() if self.image_path else False,
            'processed': self.is_processed()
        }
        
        # Añadir información de keypoints si existe
        if self._keypoint_result:
            summary['keypoints_info'] = {
                'success': self._keypoint_result.success,
                'processing_time': self._keypoint_result.processing_time,
                'has_keypoints': self._keypoint_result.keypoints is not None,
                'keypoints_shape': self._keypoint_result.keypoints.shape if self._keypoint_result.keypoints is not None else None,
                'has_scores': self._keypoint_result.scores is not None,
                'has_bbox': self._keypoint_result.bbox is not None,
                'error_message': self._keypoint_result.error_message
            }
        
        return summary
    
    def __str__(self) -> str:
        return f"Photo({self.unique_id}, processed={self.is_processed()})"
    
    def __repr__(self) -> str:
        return (f"Photo(patient_id='{self.patient_id}', session_id='{self.session_id}', "
                f"camera_id={self.camera_id}, frame_number={self.frame_number}, "
                f"chunk_number={self.chunk_number}, model_name='{self.model_name}')")

"""
Clase Video para procesamiento de chunks de video con múltiples modelos MMPose
Maneja la extracción de frames y su procesamiento sincronizado
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .photo import Photo

logger = logging.getLogger(__name__)

@dataclass
class VideoProcessingResult:
    """Resultado del procesamiento de un video"""
    success: bool
    frames_extracted: int
    frames_processed: int
    processing_time: float
    photos: List[Photo]
    errors: List[str]

class Video:
    """
    Maneja el procesamiento de un chunk de video completo
    
    Atributos:
        patient_id: ID del paciente
        session_id: ID de la sesión  
        camera_id: ID de la cámara
        chunk_number: Número del chunk
        video_path: Ruta al archivo de video
        target_fps: FPS objetivo para extracción
        models_to_use: Lista de modelos MMPose a aplicar
    """
    
    def __init__(self, patient_id: str, session_id: str, camera_id: int, 
                 chunk_number: int, video_path: Path, target_fps: int = 15,
                 models_to_use: Optional[List[str]] = None):
        self.patient_id = patient_id
        self.session_id = session_id
        self.camera_id = camera_id
        self.chunk_number = chunk_number
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.models_to_use = models_to_use or []
        
        # Validaciones
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        if self.camera_id < 0:
            raise ValueError("camera_id must be non-negative")
        if self.chunk_number < 0:
            raise ValueError("chunk_number must be non-negative")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
    
    @property
    def unique_id(self) -> str:
        """ID único para este video"""
        return f"p{self.patient_id}_s{self.session_id}_c{self.camera_id}_ch{self.chunk_number}"
    
    def get_video_info(self) -> Dict[str, Any]:
        """
        Obtener información del video
        
        Returns:
            Diccionario con metadatos del video
        """
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            
            if not cap.isOpened():
                raise ValueError(f"No se puede abrir el video: {self.video_path}")
            
            info = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'original_fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration_seconds': 0,
                'file_size_mb': self.video_path.stat().st_size / (1024 * 1024)
            }
            
            if info['original_fps'] > 0:
                info['duration_seconds'] = info['total_frames'] / info['original_fps']
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo info del video {self.video_path}: {e}")
            return {}
    
    def extract_frames(self) -> List[Tuple[int, np.ndarray]]:
        """
        Extraer frames del video a FPS objetivo
        
        Returns:
            Lista de tuplas (frame_number, image)
        """
        extracted_frames = []
        
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            
            if not cap.isOpened():
                raise ValueError(f"No se puede abrir el video: {self.video_path}")
            
            video_info = self.get_video_info()
            original_fps = video_info.get('original_fps', 30)
            total_frames = video_info.get('total_frames', 0)
            
            if original_fps <= 0:
                logger.error(f"FPS inválido en video: {original_fps}")
                cap.release()
                return extracted_frames
            
            # Calcular intervalo de frames para FPS objetivo
            frame_interval = max(1, int(original_fps / self.target_fps))
            
            logger.info(f"Extrayendo frames de {self.unique_id}: "
                       f"FPS original: {original_fps}, FPS objetivo: {self.target_fps}, "
                       f"Intervalo: cada {frame_interval} frames")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extraer frame si corresponde al intervalo
                if frame_count % frame_interval == 0:
                    extracted_frames.append((extracted_count, frame.copy()))
                    extracted_count += 1
                    
                    # Límite de seguridad
                    if extracted_count >= 100:  # Máximo 100 frames por chunk
                        logger.warning(f"Límite de frames alcanzado para {self.unique_id}")
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Frames extraídos de {self.unique_id}: {len(extracted_frames)}")
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Error extrayendo frames de {self.video_path}: {e}")
            return []
    
    def save_frames_as_photos(self, frames: List[Tuple[int, np.ndarray]]) -> List[Photo]:
        """
        Guardar frames extraídos como archivos de imagen y crear objetos Photo
        
        Args:
            frames: Lista de tuplas (frame_number, image)
            
        Returns:
            Lista de objetos Photo creados
        """
        photos = []
        
        try:
            for frame_number, image in frames:
                # Crear Photo para cada modelo que se va a usar
                for model_name in self.models_to_use:
                    photo = Photo(
                        patient_id=self.patient_id,
                        session_id=self.session_id,
                        camera_id=self.camera_id,
                        frame_number=frame_number,
                        chunk_number=self.chunk_number,
                        model_name=model_name
                    )
                    
                    # Guardar imagen solo una vez (usar el primer modelo)
                    if model_name == self.models_to_use[0]:
                        if photo.save_image(image):
                            # Asignar la misma ruta a todos los objetos Photo del mismo frame
                            image_path = photo.expected_image_path
                        else:
                            logger.error(f"Error guardando imagen para frame {frame_number}")
                            continue
                    else:
                        # Usar la misma imagen para otros modelos del mismo frame
                        image_path = Photo(
                            patient_id=self.patient_id,
                            session_id=self.session_id,
                            camera_id=self.camera_id,
                            frame_number=frame_number,
                            chunk_number=self.chunk_number,
                            model_name=self.models_to_use[0]
                        ).expected_image_path
                    
                    photo.image_path = image_path
                    photos.append(photo)
            
            logger.info(f"Creados {len(photos)} objetos Photo para {self.unique_id}")
            return photos
            
        except Exception as e:
            logger.error(f"Error guardando frames como fotos: {e}")
            return []
    
    def process_with_mmpose_models(self, photos: List[Photo], 
                                  inferencers: Dict[str, Any]) -> VideoProcessingResult:
        """
        Procesar todas las fotos con sus respectivos modelos MMPose
        
        Args:
            photos: Lista de objetos Photo a procesar
            inferencers: Diccionario {model_name: inferencer}
            
        Returns:
            Resultado del procesamiento completo
        """
        start_time = time.time()
        processed_count = 0
        errors = []
        
        try:
            # Agrupar fotos por modelo para procesamiento eficiente
            photos_by_model = {}
            for photo in photos:
                if photo.model_name not in photos_by_model:
                    photos_by_model[photo.model_name] = []
                photos_by_model[photo.model_name].append(photo)
            
            # Procesar cada modelo
            for model_name, model_photos in photos_by_model.items():
                if model_name not in inferencers:
                    error_msg = f"Inferencer no disponible para modelo: {model_name}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                inferencer = inferencers[model_name]
                
                logger.info(f"Procesando {len(model_photos)} fotos con modelo {model_name}")
                
                # Procesar fotos del modelo
                for photo in model_photos:
                    try:
                        if photo.process_with_mmpose(inferencer):
                            if photo.save_keypoints():
                                processed_count += 1
                            else:
                                errors.append(f"Error guardando keypoints: {photo.unique_id}")
                        else:
                            errors.append(f"Error procesando con MMPose: {photo.unique_id}")
                            
                    except Exception as e:
                        error_msg = f"Error procesando {photo.unique_id}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
            
            processing_time = time.time() - start_time
            
            # Crear resultado
            result = VideoProcessingResult(
                success=processed_count > 0 and len(errors) == 0,
                frames_extracted=len(set(p.frame_number for p in photos)),
                frames_processed=processed_count,
                processing_time=processing_time,
                photos=photos,
                errors=errors
            )
            
            logger.info(f"Procesamiento completado para {self.unique_id}: "
                       f"{processed_count} fotos procesadas en {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error en procesamiento de video {self.unique_id}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return VideoProcessingResult(
                success=False,
                frames_extracted=0,
                frames_processed=processed_count,
                processing_time=time.time() - start_time,
                photos=photos,
                errors=errors
            )
    
    def process_complete_pipeline(self, inferencers: Dict[str, Any]) -> VideoProcessingResult:
        """
        Ejecutar pipeline completo: extracción + procesamiento + guardado
        
        Args:
            inferencers: Diccionario {model_name: inferencer}
            
        Returns:
            Resultado del procesamiento completo
        """
        logger.info(f"🎬 Iniciando procesamiento completo de {self.unique_id}")
        
        try:
            # 1. Extraer frames
            frames = self.extract_frames()
            if not frames:
                return VideoProcessingResult(
                    success=False,
                    frames_extracted=0,
                    frames_processed=0,
                    processing_time=0,
                    photos=[],
                    errors=["No se pudieron extraer frames del video"]
                )
            
            # 2. Guardar frames como fotos
            photos = self.save_frames_as_photos(frames)
            if not photos:
                return VideoProcessingResult(
                    success=False,
                    frames_extracted=len(frames),
                    frames_processed=0,
                    processing_time=0,
                    photos=[],
                    errors=["No se pudieron guardar frames como fotos"]
                )
            
            # 3. Procesar con MMPose
            result = self.process_with_mmpose_models(photos, inferencers)
            
            logger.info(f"✅ Pipeline completado para {self.unique_id}: "
                       f"Éxito: {result.success}, Frames: {result.frames_extracted}, "
                       f"Procesados: {result.frames_processed}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error en pipeline completo {self.unique_id}: {e}"
            logger.error(error_msg)
            
            return VideoProcessingResult(
                success=False,
                frames_extracted=0,
                frames_processed=0,
                processing_time=0,
                photos=[],
                errors=[error_msg]
            )
    
    def cleanup_temp_images(self):
        """Limpiar imágenes temporales después del procesamiento"""
        try:
            from config import data_config
            
            photos_dir = (data_config.photos_dir / 
                         f"patient{self.patient_id}" / 
                         f"session{self.session_id}" / 
                         f"camera{self.camera_id}")
            
            if photos_dir.exists():
                import shutil
                shutil.rmtree(photos_dir)
                logger.info(f"🗑️ Imágenes temporales eliminadas: {photos_dir}")
                
        except Exception as e:
            logger.error(f"Error limpiando imágenes temporales: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen del video"""
        video_info = self.get_video_info()
        
        return {
            'unique_id': self.unique_id,
            'patient_id': self.patient_id,
            'session_id': self.session_id,
            'camera_id': self.camera_id,
            'chunk_number': self.chunk_number,
            'video_path': str(self.video_path),
            'target_fps': self.target_fps,
            'models_to_use': self.models_to_use,
            'video_exists': self.video_path.exists(),
            'video_info': video_info
        }

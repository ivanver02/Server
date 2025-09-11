import numpy as np
from pathlib import Path
from typing import Optional
import logging
from abc import ABC

from mmpose.apis import MMPoseInferencer
from config import mmpose_config, data_config, processing_config

logger = logging.getLogger(__name__)


class BasePoseDetector(ABC):
    """
    Clase base para todos los detectores de pose
    """
    
    def __init__(self, model_name: str, config_key: str):
        """
        Inicializar detector base
        
        Args:
            model_name: Nombre del modelo (usado para directorios)
            config_key: Clave en mmpose_config para obtener la configuración
        """
        self.model_name = model_name.lower()
        self.config_key = config_key
        self.inferencer: Optional[MMPoseInferencer] = None
        self.config = getattr(mmpose_config, config_key)
        self.is_initialized = False
        self.keypoints_names = None  # Será asignado por cada detector específico
        self.final_keypoints_idx = None  # Correspondencia entre los índices de los keypoints en el orden que se van a guardar finalmente (almacenado en EnsembleProcessor) con respecto a los de este detector, con la estructura {posicion_final_keypoint_name:posicion_keypoint_detector}
        
    def initialize(self) -> bool:
        """
        Inicializar el detector con manejo de errores usando URLs directas
        """
        try:
            # Obtener URLs directas desde la configuración
            pose2d_url = self.config['pose2d']
            pose2d_weights_url = self.config['pose2d_weights']
            
            # Verificar que las URLs sean válidas
            if not pose2d_url.startswith('http'):
                logger.error(f"Invalid config URL for {self.model_name}: {pose2d_url}")
                return False
                
            if not pose2d_weights_url.startswith('http'):
                logger.error(f"Invalid weights URL for {self.model_name}: {pose2d_weights_url}")
                return False
            
            # Inicializar el inferenciador directamente con las URLs
            logger.info(f"Initializing {self.model_name} with config URL: {pose2d_url}")
            self.inferencer = MMPoseInferencer(
                pose2d=pose2d_url,
                pose2d_weights=pose2d_weights_url,
                device=self.config['device']
            )
            
            self.is_initialized = True
            logger.info(f"{self.model_name} detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.model_name} detector: {e}")
            return False
    
    def process_chunk(self, video_path: Path, patient_id: str, session_id: str, 
                     camera_id: int, chunk_id: str) -> bool:
        """
        Procesar un chunk de video y guardar keypoints
        
        Args:
            video_path: Ruta al archivo de video del chunk
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_id: ID del chunk
            
        Returns:
            True si el procesamiento fue exitoso
        """
        if not self.is_initialized:
            logger.error("Detector not initialized")
            return False
            
        try:
            # Crear directorios de salida para keypoints
            keypoints_base_dir = self._create_keypoints_directories(
                patient_id, session_id, camera_id
            )

            # Configurar argumentos iniciales, sin guardar vídeo anotado, para el inferenciador
            inference_args = {
                'return_vis': processing_config.save_annotated_videos,
                'show': False
            }
            
            # Si hay que guardar videos anotados, configurar directorios
            if processing_config.save_annotated_videos:
                annotated_dir = self._create_annotated_video_directory(
                    patient_id, session_id, camera_id
                )
                inference_args.update({
                    'vis_out_dir': str(annotated_dir),
                    'video_out': str(annotated_dir / f'{chunk_id}_annotated.mp4'),
                    'radius': 4,
                    'thickness': 2,
                    'skeleton_style': 'mmpose'
                })
            
            # Ejecutar inferencia
            result_generator = self.inferencer(str(video_path), **inference_args)
            
            # Procesar resultados frame por frame
            coordinates_dir = keypoints_base_dir / 'coordinates'
            confidence_dir = keypoints_base_dir / 'confidence'
            
            for frame_idx, results in enumerate(result_generator):
                if results['predictions'] and len(results['predictions'][0]) > 0:
                    # Extraer keypoints y confianzas
                    keypoints = np.array(results['predictions'][0][0]['keypoints'])
                    scores = np.array(results['predictions'][0][0]['keypoint_scores'])
                    
                    # Guardar archivos
                    filename = f"{frame_idx}_{chunk_id}"
                    np.save(coordinates_dir / f"{filename}.npy", keypoints)
                    np.save(confidence_dir / f"{filename}.npy", scores)
                else:
                    # Si no hay detecciones, guardar arrays vacíos
                    filename = f"{frame_idx}_{chunk_id}"
                    num_keypoints = len(self.keypoints_names) if self.keypoints_names else 17
                    empty_keypoints = np.zeros((num_keypoints, 2))  # N keypoints, 2 coordenadas
                    empty_scores = np.zeros(num_keypoints)  # N scores
                    np.save(coordinates_dir / f"{filename}.npy", empty_keypoints)
                    np.save(confidence_dir / f"{filename}.npy", empty_scores)
            
            logger.info(f"{self.model_name} processing completed for chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            return False
    
    def _create_keypoints_directories(self, patient_id: str, session_id: str, 
                                    camera_id: int) -> Path:
        """Crear directorios para guardar keypoints"""
        base_dir = (data_config.unprocessed_dir / f"patient{patient_id}" / 
                   f"session{session_id}" / "keypoints2D" / self.model_name / 
                   f"camera{camera_id}")
        
        coordinates_dir = base_dir / 'coordinates'
        confidence_dir = base_dir / 'confidence'
        
        coordinates_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)
        
        return base_dir
    
    def _create_annotated_video_directory(self, patient_id: str, session_id: str, 
                                        camera_id: int) -> Path:
        """Crear directorio para videos anotados"""
        annotated_dir = (data_config.annotated_videos_dir / self.model_name / 
                        f"patient{patient_id}" / f"session{session_id}" / 
                        f"camera{camera_id}")
        
        annotated_dir.mkdir(parents=True, exist_ok=True)
        return annotated_dir

"""
Procesador de video principal
Coordina la extracción de frames, procesamiento con MMPose y ensemble learning
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import time

from .models.video import Video, VideoProcessingResult
from .pose_detector import mmpose_wrapper
from .ensemble_processor import ensemble_processor

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Procesador principal de video que coordina todo el pipeline
    """
    
    def __init__(self):
        self.is_initialized = False
        self.processing_queue = []
        self.max_concurrent_videos = 2  # Máximo videos procesando simultáneamente
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def initialize(self) -> bool:
        """
        Inicializar el procesador de video
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            logger.info("Inicializando VideoProcessor...")
            
            # Inicializar wrapper de MMPose
            if not mmpose_wrapper.initialize_models():
                logger.error("Error inicializando modelos MMPose")
                return False
            
            self.is_initialized = True
            logger.info("VideoProcessor inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando VideoProcessor: {e}")
            return False
    
    def process_video_chunk(self, patient_id: str, session_id: str, 
                           camera_id: int, chunk_number: int, 
                           video_path: Path) -> VideoProcessingResult:
        """
        Procesar un chunk de video completo
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_number: Número del chunk
            video_path: Ruta al archivo de video
            
        Returns:
            Resultado del procesamiento
        """
        if not self.is_initialized:
            logger.error("VideoProcessor no inicializado")
            return VideoProcessingResult(
                success=False,
                frames_extracted=0,
                frames_processed=0,
                processing_time=0,
                photos=[],
                errors=["VideoProcessor no inicializado"]
            )
        
        try:
            logger.info(f"🎬 Procesando chunk {chunk_number} - "
                       f"Paciente: {patient_id}, Sesión: {session_id}, Cámara: {camera_id}")
            
            start_time = time.time()
            
            # 1. Obtener modelos a usar
            from config import processing_config
            models_to_use = processing_config.coco_models + processing_config.extended_models
            
            # 2. Crear objeto Video
            video = Video(
                patient_id=patient_id,
                session_id=session_id,
                camera_id=camera_id,
                chunk_number=chunk_number,
                video_path=video_path,
                target_fps=processing_config.target_fps,
                models_to_use=models_to_use
            )
            
            # 3. Obtener inferencers
            inferencers = mmpose_wrapper.get_all_inferencers()
            
            # Verificar que todos los modelos están disponibles
            missing_models = [m for m in models_to_use if m not in inferencers]
            if missing_models:
                logger.warning(f"Modelos no disponibles: {missing_models}")
                # Continuar con los modelos disponibles
                available_models = [m for m in models_to_use if m in inferencers]
                video.models_to_use = available_models
                
                if not available_models:
                    return VideoProcessingResult(
                        success=False,
                        frames_extracted=0,
                        frames_processed=0,
                        processing_time=time.time() - start_time,
                        photos=[],
                        errors=["No hay modelos MMPose disponibles"]
                    )
            
            # 4. Procesar video completo
            result = video.process_complete_pipeline(inferencers)
            
            # 5. Limpiar imágenes temporales
            video.cleanup_temp_images()
            
            total_time = time.time() - start_time
            result.processing_time = total_time
            
            if result.success:
                logger.info(f"Chunk {chunk_number} procesado exitosamente en {total_time:.2f}s - "
                           f"Frames: {result.frames_extracted}, Procesados: {result.frames_processed}")
            else:
                logger.error(f"Error procesando chunk {chunk_number}: {result.errors}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error crítico procesando chunk {chunk_number}: {e}"
            logger.error(error_msg)
            
            return VideoProcessingResult(
                success=False,
                frames_extracted=0,
                frames_processed=0,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0,
                photos=[],
                errors=[error_msg]
            )
    
    def process_and_ensemble_chunk(self, patient_id: str, session_id: str,
                                  camera_id: int, chunk_number: int,
                                  video_path: Path) -> Dict[str, Any]:
        """
        Procesar chunk y aplicar ensemble learning
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_number: Número del chunk
            video_path: Ruta al archivo de video
            
        Returns:
            Resumen completo del procesamiento
        """
        try:
            logger.info(f"🔄 Procesamiento completo chunk {chunk_number} - C{camera_id}")
            
            # 1. Procesar video con MMPose
            video_result = self.process_video_chunk(
                patient_id, session_id, camera_id, chunk_number, video_path
            )
            
            if not video_result.success:
                return {
                    'success': False,
                    'stage': 'video_processing',
                    'video_result': video_result.__dict__,
                    'ensemble_result': None
                }
            
            # 2. Aplicar ensemble learning a los frames procesados
            logger.info(f"🎯 Aplicando ensemble learning para chunk {chunk_number}")
            
            # Obtener números de frame únicos
            frame_numbers = list(set(photo.frame_number for photo in video_result.photos))
            
            ensemble_results = []
            ensemble_errors = []
            
            for frame_num in frame_numbers:
                # Calcular número global de frame
                global_frame = chunk_number * 75 + frame_num  # Asumiendo ~75 frames por chunk
                
                # Procesar ensemble para este frame
                ensemble_result = ensemble_processor.process_frame_ensemble(
                    patient_id, session_id, camera_id, global_frame
                )
                
                if ensemble_result:
                    # Guardar resultado del ensemble
                    if ensemble_processor.save_ensemble_result(
                        ensemble_result, patient_id, session_id, camera_id, global_frame
                    ):
                        ensemble_results.append(ensemble_result.processing_info)
                    else:
                        ensemble_errors.append(f"Error guardando ensemble frame {global_frame}")
                else:
                    ensemble_errors.append(f"Error en ensemble frame {global_frame}")
            
            success = len(ensemble_errors) == 0 and len(ensemble_results) > 0
            
            result = {
                'success': success,
                'video_result': {
                    'frames_extracted': video_result.frames_extracted,
                    'frames_processed': video_result.frames_processed,
                    'processing_time': video_result.processing_time
                },
                'ensemble_result': {
                    'frames_ensembled': len(ensemble_results),
                    'ensemble_errors': ensemble_errors,
                    'mean_confidence': sum(r['mean_confidence'] for r in ensemble_results) / len(ensemble_results) if ensemble_results else 0
                },
                'chunk_info': {
                    'patient_id': patient_id,
                    'session_id': session_id,
                    'camera_id': camera_id,
                    'chunk_number': chunk_number
                }
            }
            
            if success:
                logger.info(f"Procesamiento completo exitoso chunk {chunk_number} - "
                           f"Ensemble: {len(ensemble_results)} frames")
            else:
                logger.error(f"Errores en procesamiento chunk {chunk_number}: {ensemble_errors}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error en procesamiento completo chunk {chunk_number}: {e}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'video_result': None,
                'ensemble_result': None
            }
    
    async def process_chunk_async(self, patient_id: str, session_id: str,
                                 camera_id: int, chunk_number: int,
                                 video_path: Path) -> Dict[str, Any]:
        """
        Procesar chunk de forma asíncrona
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_number: Número del chunk
            video_path: Ruta al archivo de video
            
        Returns:
            Resultado del procesamiento
        """
        loop = asyncio.get_event_loop()
        
        # Ejecutar procesamiento en thread pool
        result = await loop.run_in_executor(
            self.executor,
            self.process_and_ensemble_chunk,
            patient_id, session_id, camera_id, chunk_number, video_path
        )
        
        return result
    
    def queue_chunk_for_processing(self, patient_id: str, session_id: str,
                                  camera_id: int, chunk_number: int,
                                  video_path: Path) -> str:
        """
        Añadir chunk a la cola de procesamiento
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_number: Número del chunk
            video_path: Ruta al archivo de video
            
        Returns:
            ID de la tarea en cola
        """
        task_id = f"{patient_id}_{session_id}_{camera_id}_{chunk_number}"
        
        task_info = {
            'task_id': task_id,
            'patient_id': patient_id,
            'session_id': session_id,
            'camera_id': camera_id,
            'chunk_number': chunk_number,
            'video_path': video_path,
            'status': 'queued',
            'created_at': time.time()
        }
        
        self.processing_queue.append(task_info)
        
        logger.info(f"📝 Chunk añadido a cola: {task_id}")
        
        return task_id
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Obtener estado de la cola de procesamiento"""
        return {
            'queue_length': len(self.processing_queue),
            'tasks': [
                {
                    'task_id': task['task_id'],
                    'status': task['status'],
                    'created_at': task['created_at']
                }
                for task in self.processing_queue
            ]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema"""
        return {
            'initialized': self.is_initialized,
            'mmpose_status': mmpose_wrapper.get_status(),
            'queue_status': self.get_queue_status(),
            'max_concurrent_videos': self.max_concurrent_videos
        }
    
    def cleanup(self):
        """Limpiar recursos del procesador"""
        try:
            logger.info("🧹 Limpiando VideoProcessor...")
            
            # Limpiar MMPose
            mmpose_wrapper.cleanup()
            
            # Cerrar executor
            self.executor.shutdown(wait=True)
            
            # Limpiar cola
            self.processing_queue.clear()
            
            self.is_initialized = False
            
            logger.info("VideoProcessor limpiado")
            
        except Exception as e:
            logger.error(f"Error limpiando VideoProcessor: {e}")

# Instancia global del procesador
video_processor = VideoProcessor()

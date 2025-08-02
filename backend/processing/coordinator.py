import logging
import threading
import os
from pathlib import Path
from typing import Dict, Any

from .detectors import VitPoseDetector, HRNetDetector, CSPDetector
# from .detectors import MSPNDetector  # Comentado - no usar MSPN
from config.settings import gpu_config

logger = logging.getLogger(__name__)


class PoseProcessingCoordinator:
    """
    Coordinador simple para ejecutar múltiples detectores de pose con soporte multi-GPU configurable
    """
    
    def __init__(self):
        """Inicializar coordinador con todos los detectores disponibles"""
        self.detectors = [
            VitPoseDetector(),
            # MSPNDetector(),
            HRNetDetector(),
            CSPDetector()
        ]
        self.initialized = False
        
        # Sistema de asignación de GPUs basado en configuración
        self.gpu_lock = threading.Lock()
        self.gpu_usage = gpu_config.get_gpu_usage_dict()  # Usar GPUs configuradas
        self.available_gpus = gpu_config.available_gpus.copy()
        
        logger.info(f"GPUs configuradas: {self.available_gpus}")
        logger.info(f"Máximo chunks concurrentes: {gpu_config.max_concurrent_chunks}")
    
    def initialize_all(self) -> bool:
        """
        Inicializar todos los detectores
        
        Returns:
            True si al menos uno se inicializó correctamente
        """
        success_count = 0
        
        for detector in self.detectors:
            try:
                if detector.initialize():
                    success_count += 1
                    logger.info(f"Detector {detector.model_name} initialized successfully")
                else:
                    logger.warning(f"Failed to initialize detector {detector.model_name}")
            except Exception as e:
                logger.error(f"Error initializing detector {detector.model_name}: {e}")
        
        self.initialized = success_count > 0
        logger.info(f"Pose coordinator initialized: {success_count}/{len(self.detectors)} detectors ready")
        
        return self.initialized
    
    def _get_available_gpu(self) -> int:
        """
        Obtener una GPU disponible de manera thread-safe basado en configuración
        
        Returns:
            ID de GPU disponible según configuración, o -1 si no hay GPUs disponibles
        """
        with self.gpu_lock:
            # Si no hay GPUs configuradas, retornar -1 (usar CPU)
            if not self.available_gpus:
                logger.debug("Configuración: Solo CPU (sin GPUs configuradas)")
                return -1
                
            for gpu_id in self.available_gpus:
                if gpu_id in self.gpu_usage and not self.gpu_usage[gpu_id]:
                    self.gpu_usage[gpu_id] = True
                    logger.debug(f"GPU {gpu_id} ASIGNADA")
                    return gpu_id
            
            logger.warning(f"Todas las GPUs están ocupadas {self.available_gpus} - esperando liberación")
            return -1  # No hay GPUs disponibles
    
    def _release_gpu(self, gpu_id: int):
        """
        Liberar una GPU de manera thread-safe
        
        Args:
            gpu_id: ID de la GPU a liberar
        """
        with self.gpu_lock:
            if gpu_id in self.gpu_usage:
                self.gpu_usage[gpu_id] = False
                logger.debug(f"GPU {gpu_id} LIBERADA ")
            
    def _set_gpu_for_processing(self, gpu_id: int):
        """
        Configurar la GPU para el procesamiento actual
        
        Args:
            gpu_id: ID de la GPU a usar (0 o 1)
        """
        if gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.debug(f"CUDA_VISIBLE_DEVICES configurado a GPU {gpu_id}")
        else:
            logger.warning(" No se pudo asignar GPU, usando CPU o GPU por defecto")
    
    def process_chunk(self, video_path: Path, patient_id: str, session_id: str, 
                     camera_id: int, chunk_id: str) -> Dict[str, bool]:
        """
        Procesar un chunk con todos los detectores inicializados usando asignación automática de GPU
        
        Args:
            video_path: Ruta al archivo de video del chunk
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_id: ID del chunk
            
        Returns:
            Diccionario con el resultado de cada detector {detector_name: success}
        """
        if not self.initialized:
            logger.error("Coordinator not initialized")
            return {}

        # Asignar una GPU para este procesamiento
        assigned_gpu = self._get_available_gpu()
        
        # Mensaje inicial claro sobre qué GPU se está usando
        if assigned_gpu >= 0:
            logger.info(f"INICIANDO procesamiento de Cámara {camera_id} - Chunk {chunk_id} en GPU {assigned_gpu}")
        else:
            logger.info(f"INICIANDO procesamiento de Cámara {camera_id} - Chunk {chunk_id} en CPU (sin GPU disponible)")
        
        try:
            # Configurar GPU para el procesamiento
            self._set_gpu_for_processing(assigned_gpu)
            
            results = {}
            
            for detector in self.detectors:
                if detector.is_initialized:
                    try:
                        # Mensaje específico por detector indicando GPU
                        gpu_info = f"GPU {assigned_gpu}" if assigned_gpu >= 0 else "CPU"
                        logger.info(f"Procesando con {detector.model_name} en {gpu_info} | Cámara {camera_id} - Chunk {chunk_id}")
                        
                        success = detector.process_chunk(
                            video_path=video_path,
                            patient_id=patient_id,
                            session_id=session_id,
                            camera_id=camera_id,
                            chunk_id=chunk_id
                        )
                        results[detector.model_name] = success
                        
                        if success:
                            logger.info(f"{detector.model_name} completado en {gpu_info} | Cámara {camera_id} - Chunk {chunk_id}")
                        else:
                            logger.warning(f"{detector.model_name} falló en {gpu_info} | Cámara {camera_id} - Chunk {chunk_id}")
                            
                    except Exception as e:
                        gpu_info = f"GPU {assigned_gpu}" if assigned_gpu >= 0 else "CPU"
                        logger.error(f"Error en {detector.model_name} en {gpu_info} | Cámara {camera_id} - Chunk {chunk_id}: {e}")
                        results[detector.model_name] = False
                else:
                    logger.debug(f"Saltando {detector.model_name} - no inicializado")
                    results[detector.model_name] = False

            success_count = sum(results.values())
            gpu_info = f"GPU {assigned_gpu}" if assigned_gpu >= 0 else "CPU"
            logger.info(f"COMPLETADO procesamiento en {gpu_info} | Cámara {camera_id} - Chunk {chunk_id}: {success_count}/{len(results)} detectores exitosos")
            
            return results
            
        finally:
            # Liberar la GPU al finalizar
            if assigned_gpu >= 0:
                self._release_gpu(assigned_gpu)
    
    # Se usa en el endpoint /api/gpu/status
    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Obtener el estado actual de las GPUs basado en configuración
        
        Returns:
            Diccionario con información del estado de las GPUs configuradas
        """
        with self.gpu_lock:
            gpu_status = {}
            for gpu_id in self.available_gpus:
                status = "ocupada" if self.gpu_usage.get(gpu_id, False) else "libre"
                gpu_status[f'gpu_{gpu_id}'] = status
            
            available_count = sum(1 for gpu_id in self.available_gpus 
                                if not self.gpu_usage.get(gpu_id, False))
            
            return {
                'configured_gpus': self.available_gpus,
                'gpu_status': gpu_status,
                'available_gpus': available_count,
                'total_gpus': len(self.available_gpus),
                'max_concurrent_chunks': gpu_config.max_concurrent_chunks,
                'mode': f"multi-gpu" if len(self.available_gpus) > 1 else f"single-gpu" if len(self.available_gpus) == 1 else "cpu-only"
            }
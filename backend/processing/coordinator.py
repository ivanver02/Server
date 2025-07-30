"""
Coordinador para el procesamiento de pose con múltiples detectores
"""
import logging
from pathlib import Path
from typing import List, Dict, Any

from .detectors import VitPoseDetector, MSPNDetector, HRNetDetector, CSPDetector

logger = logging.getLogger(__name__)


class PoseProcessingCoordinator:
    """
    Coordinador simple para ejecutar múltiples detectores de pose
    """
    
    def __init__(self):
        """Inicializar coordinador con todos los detectores disponibles"""
        self.detectors = [
            VitPoseDetector(),
            MSPNDetector(), 
            HRNetDetector(),
            CSPDetector()
        ]
        self.initialized = False
    
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
    
    def process_chunk(self, video_path: Path, patient_id: str, session_id: str, 
                     camera_id: int, chunk_id: str) -> Dict[str, bool]:
        """
        Procesar un chunk con todos los detectores inicializados
        
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
        
        results = {}
        
        for detector in self.detectors:
            if detector.is_initialized:
                try:
                    logger.info(f"🔍 Analizando chunk {chunk_id} de cámara {camera_id} con detector {detector.model_name}")
                    
                    success = detector.process_chunk(
                        video_path=video_path,
                        patient_id=patient_id,
                        session_id=session_id,
                        camera_id=camera_id,
                        chunk_id=chunk_id
                    )
                    results[detector.model_name] = success
                    
                    if success:
                        logger.info(f"✅ Chunk {chunk_id} cámara {camera_id} procesado exitosamente con {detector.model_name}")
                    else:
                        logger.warning(f"❌ Falló el procesamiento del chunk {chunk_id} cámara {camera_id} con {detector.model_name}")
                        
                except Exception as e:
                    logger.error(f"💥 Error procesando chunk {chunk_id} cámara {camera_id} con {detector.model_name}: {e}")
                    results[detector.model_name] = False
            else:
                logger.debug(f"⏭️  Saltando {detector.model_name} - no inicializado")
                results[detector.model_name] = False
        
        success_count = sum(results.values())
        logger.info(f"🎬 Chunk {chunk_id} cámara {camera_id} procesamiento completado: {success_count}/{len(results)} detectores exitosos")
        
        return results

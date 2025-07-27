"""
Manager para gestionar múltiples detectores MMPose
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..base import BaseDetectorManager, BasePoseDetector
from .detector import MMPoseDetector

logger = logging.getLogger(__name__)


class MMPoseManager(BaseDetectorManager):
    """
    Manager específico para detectores MMPose
    Maneja múltiples modelos MMPose con diferentes configuraciones
    """
    
    def __init__(self, models_base_path: Optional[Path] = None):
        super().__init__()
        self.models_base_path = models_base_path
        self.detector_type = 'mmpose'
    
    def register_mmpose_model(self, model_name: str, 
                             model_path: Optional[Path] = None) -> bool:
        """
        Registrar un modelo MMPose específico
        
        Args:
            model_name: Nombre del modelo
            model_path: Ruta al directorio del modelo (con .py y .pth)
            
        Returns:
            True si se registró correctamente
        """
        try:
            # Si no se proporciona path, usar el path base + nombre del modelo
            if model_path is None and self.models_base_path:
                model_path = self.models_base_path / model_name
            
            if model_path and not model_path.exists():
                logger.error(f"Ruta del modelo no existe: {model_path}")
                return False
            
            # Crear detector MMPose
            detector = MMPoseDetector(model_name, model_path)
            
            return self.register_detector(detector)
            
        except Exception as e:
            logger.error(f"Error registrando modelo MMPose {model_name}: {e}")
            return False
    
    def register_detector(self, detector: BasePoseDetector) -> bool:
        """
        Registrar un detector en el manager
        """
        try:
            if not isinstance(detector, MMPoseDetector):
                logger.error("Solo se pueden registrar detectores MMPose en este manager")
                return False
            
            if detector.model_name in self.detectors:
                logger.warning(f"Detector {detector.model_name} ya existe, se reemplazará")
            
            self.detectors[detector.model_name] = detector
            logger.info(f"Detector MMPose registrado: {detector.model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registrando detector: {e}")
            return False
    
    def initialize_all(self) -> bool:
        """
        Inicializar todos los detectores registrados
        """
        if not self.detectors:
            logger.warning("No hay detectores registrados para inicializar")
            return False
        
        success_count = 0
        self.active_models.clear()
        
        for model_name, detector in self.detectors.items():
            try:
                if detector.initialize():
                    self.active_models.append(model_name)
                    success_count += 1
                    logger.info(f"Detector inicializado exitosamente: {model_name}")
                else:
                    logger.error(f"Error inicializando detector: {model_name}")
            
            except Exception as e:
                logger.error(f"Excepción inicializando detector {model_name}: {e}")
        
        logger.info(f"Detectores MMPose inicializados: {success_count}/{len(self.detectors)}")
        return success_count > 0
    
    def auto_discover_models(self, models_directory: Path) -> int:
        """
        Descubrir automáticamente modelos en un directorio
        
        Args:
            models_directory: Directorio que contiene carpetas de modelos
            
        Returns:
            Número de modelos descubiertos y registrados
        """
        if not models_directory.exists():
            logger.error(f"Directorio de modelos no existe: {models_directory}")
            return 0
        
        discovered_count = 0
        
        try:
            # Buscar carpetas que contengan archivos .py y .pth
            for model_dir in models_directory.iterdir():
                if model_dir.is_dir():
                    py_files = list(model_dir.glob("*.py"))
                    pth_files = list(model_dir.glob("*.pth"))
                    
                    if py_files and pth_files:
                        model_name = model_dir.name
                        if self.register_mmpose_model(model_name, model_dir):
                            discovered_count += 1
                            logger.info(f"Modelo descubierto: {model_name}")
                        else:
                            logger.warning(f"No se pudo registrar modelo: {model_name}")
            
            logger.info(f"Modelos MMPose descubiertos: {discovered_count}")
            return discovered_count
            
        except Exception as e:
            logger.error(f"Error descubriendo modelos: {e}")
            return 0
    
    def get_detector_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información detallada de un detector
        """
        detector = self.get_detector(model_name)
        if detector:
            return detector.get_model_info()
        return None
    
    def get_all_detector_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener información de todos los detectores
        """
        info = {}
        for model_name, detector in self.detectors.items():
            info[model_name] = detector.get_model_info()
        return info
    
    def is_model_active(self, model_name: str) -> bool:
        """
        Verificar si un modelo específico está activo (inicializado)
        """
        return model_name in self.active_models
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del manager
        """
        return {
            'detector_type': self.detector_type,
            'models_base_path': str(self.models_base_path) if self.models_base_path else None,
            'total_detectors': len(self.detectors),
            'active_detectors': len(self.active_models),
            'registered_models': list(self.detectors.keys()),
            'active_models': self.active_models.copy(),
            'detector_info': self.get_all_detector_info()
        }

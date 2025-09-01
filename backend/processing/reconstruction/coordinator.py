"""
Coordinador principal de reconstrucción 3D
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from .camera import CameraSystem
from .calculate_intrinsics import CameraCalibrator, calibrate_from_images
from .triangulation_svd import triangulate_with_svd
from .bundle_adjustment import optimize_with_bundle_adjustment
from .reprojection import validate_reconstruction
from config.camera_intrinsics import CAMERA_INTRINSICS

logger = logging.getLogger(__name__)

class ReconstructionCoordinator:
    """
    Coordinador principal para reconstrucción 3D de keypoints
    """
    
    def __init__(self, base_data_dir: Path):
        self.base_data_dir = Path(base_data_dir)
        self.camera_system: Optional[CameraSystem] = None
        self.reconstruction_method = "svd"  # "svd" o "bundle_adjustment"
        
        # Parámetros de reconstrucción
        self.confidence_threshold = 0.3
        self.min_cameras = 2
        self.max_reprojection_error = 5.0
        
        # Directorios
        self.calibration_dir = self.base_data_dir / "calibration"
        self.reconstruction_3d_dir = self.base_data_dir / "processed" / "3D_keypoints"
        
    def initialize_camera_system(self, camera_ids: List[int], 
                                use_calibration: bool = True) -> bool:
        """
        Inicializar sistema de cámaras
        
        Args:
            camera_ids: Lista de IDs de cámaras detectadas
            use_calibration: Si usar calibración guardada o configuración por defecto
        """
        try:
            if use_calibration and self.calibration_dir.exists():
                # Cargar calibración existente
                try:
                    self.camera_system = CameraSystem.load_system(self.calibration_dir)
                    logger.info("Sistema de cámaras cargado desde calibración guardada")
                    return True
                except Exception as e:
                    logger.warning(f"No se pudo cargar calibración guardada: {e}")
            
            # Usar configuración por defecto
            calibrator = CameraCalibrator(self.base_data_dir)
            calibrator.initialize_cameras_from_config(camera_ids)
            self.camera_system = calibrator.get_camera_system()
            
            logger.info(f"Sistema de cámaras inicializado con configuración por defecto para {camera_ids}")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando sistema de cámaras: {e}")
            return False
    
    def calibrate_cameras(self, calibration_images_dir: Path, 
                         camera_ids: List[int]) -> bool:
        """
        Calibrar sistema de cámaras usando imágenes de calibración
        """
        try:
            self.camera_system = calibrate_from_images(
                self.base_data_dir, calibration_images_dir, camera_ids
            )
            
            if self.camera_system.is_system_calibrated():
                logger.info("Sistema de cámaras calibrado exitosamente")
                return True
            else:
                logger.error("Falló la calibración del sistema de cámaras")
                return False
                
        except Exception as e:
            logger.error(f"Error en calibración: {e}")
            return False
    
    def load_keypoints_data(self, patient_id: str, session_id: str) -> Tuple[Dict, Dict]:
        """
        Cargar datos de keypoints 2D del ensemble para una sesión
        
        Returns:
            Tuple (keypoints_2d, confidences_2d) donde cada uno es:
            Dict {camera_id: {frame_key: array}}
        """
        keypoints_2d = {}
        confidences_2d = {}
        
        session_dir = self.base_data_dir / "processed" / "2D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
        
        if not session_dir.exists():
            logger.error(f"No se encontró directorio de sesión: {session_dir}")
            return keypoints_2d, confidences_2d
        
        # Buscar directorios de cámaras
        for camera_dir in session_dir.iterdir():
            if camera_dir.is_dir() and camera_dir.name.startswith('camera'):
                try:
                    camera_id = int(camera_dir.name.replace('camera', ''))
                    
                    coordinates_dir = camera_dir / "coordinates"
                    confidence_dir = camera_dir / "confidence"
                    
                    if coordinates_dir.exists() and confidence_dir.exists():
                        camera_keypoints = {}
                        camera_confidences = {}
                        
                        # Cargar archivos de coordenadas
                        for coord_file in coordinates_dir.glob("*.npy"):
                            frame_key = coord_file.stem  # formato: frame_chunk
                            coordinates = np.load(coord_file)
                            
                            # Buscar archivo de confianza correspondiente
                            conf_file = confidence_dir / coord_file.name
                            if conf_file.exists():
                                confidence = np.load(conf_file)
                                
                                camera_keypoints[frame_key] = coordinates
                                camera_confidences[frame_key] = confidence
                        
                        if camera_keypoints:
                            keypoints_2d[camera_id] = camera_keypoints
                            confidences_2d[camera_id] = camera_confidences
                            logger.debug(f"Cargados {len(camera_keypoints)} frames de cámara {camera_id}")
                
                except ValueError:
                    logger.warning(f"Nombre de directorio inválido: {camera_dir.name}")
        
        logger.info(f"Cargados keypoints de {len(keypoints_2d)} cámaras para patient{patient_id}/session{session_id}")
        return keypoints_2d, confidences_2d
    
    def reconstruct_3d(self, patient_id: str, session_id: str, 
                      method: str = "svd") -> bool:
        """
        Realizar reconstrucción 3D completa para una sesión
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            method: Método de reconstrucción ("svd" o "bundle_adjustment")
        """
        if self.camera_system is None:
            logger.error("Sistema de cámaras no inicializado")
            return False
        
        if not self.camera_system.is_system_calibrated():
            logger.error("Sistema de cámaras no calibrado")
            return False
        
        # Cargar datos de keypoints 2D
        keypoints_2d, confidences_2d = self.load_keypoints_data(patient_id, session_id)
        
        if not keypoints_2d:
            logger.error("No se pudieron cargar keypoints 2D")
            return False
        
        logger.info(f"Iniciando reconstrucción 3D con método: {method}")
        
        try:
            if method == "svd":
                # Triangulación con SVD
                points_3d_sequence = triangulate_with_svd(
                    self.camera_system,
                    keypoints_2d,
                    confidences_2d,
                    self.confidence_threshold,
                    self.min_cameras,
                    self.max_reprojection_error
                )
                
            elif method == "bundle_adjustment":
                # Primero triangulación inicial con SVD
                initial_3d = triangulate_with_svd(
                    self.camera_system,
                    keypoints_2d,
                    confidences_2d,
                    self.confidence_threshold,
                    self.min_cameras,
                    self.max_reprojection_error
                )
                
                # Extraer solo coordenadas (sin confianzas) para bundle adjustment
                initial_points_only = {frame_key: coords for frame_key, (coords, _) in initial_3d.items()}
                
                # Optimización con Bundle Adjustment
                points_3d_sequence = optimize_with_bundle_adjustment(
                    self.camera_system,
                    keypoints_2d,
                    confidences_2d,
                    initial_points_only,
                    optimize_cameras=False,
                    confidence_threshold=self.confidence_threshold
                )
                
                # Convertir formato para consistencia
                points_3d_sequence = {frame_key: (coords, np.ones(coords.shape[0])) 
                                    for frame_key, coords in points_3d_sequence.items()}
            
            else:
                logger.error(f"Método de reconstrucción no válido: {method}")
                return False
            
            # Guardar resultados 3D
            success = self._save_3d_keypoints(points_3d_sequence, patient_id, session_id)
            
            if success:
                # Validar reconstrucción
                logger.info("Validando reconstrucción...")
                
                # Extraer solo coordenadas para validación
                points_only = {frame_key: coords for frame_key, (coords, _) in points_3d_sequence.items()}
                
                validation_results = validate_reconstruction(
                    self.camera_system,
                    points_only,
                    keypoints_2d,
                    confidences_2d,
                    self.confidence_threshold,
                    save_visualizations=True,
                    output_dir=self.base_data_dir / "processed" / "validation" / f"patient{patient_id}" / f"session{session_id}"
                )
                
                logger.info(f"Reconstrucción 3D completada exitosamente para patient{patient_id}/session{session_id}")
                return True
            else:
                logger.error("Error guardando resultados de reconstrucción")
                return False
                
        except Exception as e:
            logger.error(f"Error en reconstrucción 3D: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_3d_keypoints(self, points_3d_sequence: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                          patient_id: str, session_id: str) -> bool:
        """
        Guardar keypoints 3D reconstruidos
        
        Args:
            points_3d_sequence: Dict {frame_key: (coordinates, confidences)}
        """
        try:
            output_dir = self.reconstruction_3d_dir / f"patient{patient_id}" / f"session{session_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            
            for frame_key, (coordinates_3d, confidences_3d) in points_3d_sequence.items():
                # Guardar coordenadas
                coord_file = output_dir / f"{frame_key}.npy"
                np.save(coord_file, coordinates_3d)
                
                # Guardar confianzas (opcional)
                conf_file = output_dir / f"{frame_key}_confidence.npy"
                np.save(conf_file, confidences_3d)
                
                saved_count += 1
            
            logger.info(f"Guardados {saved_count} frames de keypoints 3D en {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando keypoints 3D: {e}")
            return False
    
    def set_reconstruction_parameters(self, confidence_threshold: float = 0.3,
                                    min_cameras: int = 2, 
                                    max_reprojection_error: float = 5.0,
                                    method: str = "svd"):
        """Configurar parámetros de reconstrucción"""
        self.confidence_threshold = confidence_threshold
        self.min_cameras = min_cameras
        self.max_reprojection_error = max_reprojection_error
        self.reconstruction_method = method
        
        logger.info(f"Parámetros de reconstrucción: conf_threshold={confidence_threshold}, "
                   f"min_cameras={min_cameras}, max_error={max_reprojection_error}, method={method}")
    
    def get_available_sessions(self) -> List[Tuple[str, str]]:
        """Obtener lista de sesiones disponibles para reconstrucción"""
        sessions = []
        keypoints_dir = self.base_data_dir / "processed" / "2D_keypoints"
        
        if keypoints_dir.exists():
            for patient_dir in keypoints_dir.iterdir():
                if patient_dir.is_dir() and patient_dir.name.startswith('patient'):
                    patient_id = patient_dir.name.replace('patient', '')
                    
                    for session_dir in patient_dir.iterdir():
                        if session_dir.is_dir() and session_dir.name.startswith('session'):
                            session_id = session_dir.name.replace('session', '')
                            sessions.append((patient_id, session_id))
        
        return sessions


def reconstruct_patient_session(base_data_dir: Path, patient_id: str, session_id: str,
                               camera_ids: List[int], method: str = "svd",
                               use_calibration: bool = True) -> bool:
    """
    Función principal para reconstruir una sesión específica
    """
    coordinator = ReconstructionCoordinator(base_data_dir)
    
    # Inicializar sistema de cámaras
    if not coordinator.initialize_camera_system(camera_ids, use_calibration):
        return False
    
    # Realizar reconstrucción
    return coordinator.reconstruct_3d(patient_id, session_id, method)

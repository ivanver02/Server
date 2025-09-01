"""
Sistema de reproyección para validación de reconstrucción 3D
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

from .camera import CameraSystem

logger = logging.getLogger(__name__)

class ReprojectionValidator:
    """
    Validador de reconstrucción 3D mediante reproyección a cámaras 2D
    """
    
    def __init__(self, camera_system: CameraSystem):
        self.camera_system = camera_system
        self.error_threshold = 5.0  # píxeles
    
    def reproject_points(self, points_3d: np.ndarray, camera_id: int) -> Optional[np.ndarray]:
        """
        Reproyectar puntos 3D a cámara específica
        
        Args:
            points_3d: array (N, 3) con puntos 3D
            camera_id: ID de la cámara objetivo
        
        Returns:
            points_2d: array (N, 2) con coordenadas reproyectadas o None si falla
        """
        camera = self.camera_system.get_camera(camera_id)
        
        if camera is None or not camera.is_calibrated:
            logger.error(f"Cámara {camera_id} no disponible o no calibrada")
            return None
        
        if points_3d.shape[0] == 0:
            return np.array([]).reshape(0, 2)
        
        try:
            # Filtrar puntos válidos (no NaN)
            valid_mask = ~np.isnan(points_3d).any(axis=1)
            
            if not np.any(valid_mask):
                return np.full((points_3d.shape[0], 2), np.nan)
            
            # Reproyectar solo puntos válidos
            valid_3d = points_3d[valid_mask]
            reprojected_valid = camera.project_3d_to_2d(valid_3d)
            
            # Reconstruir array completo
            reprojected_2d = np.full((points_3d.shape[0], 2), np.nan)
            reprojected_2d[valid_mask] = reprojected_valid
            
            return reprojected_2d
            
        except Exception as e:
            logger.error(f"Error en reproyección para cámara {camera_id}: {e}")
            return None
    
    def calculate_reprojection_errors(self, points_3d: np.ndarray, 
                                    observed_2d: np.ndarray,
                                    camera_id: int,
                                    confidence_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcular errores de reproyección entre puntos 3D y observaciones 2D
        
        Args:
            points_3d: array (N, 3) con puntos 3D
            observed_2d: array (N, 2) con observaciones 2D
            camera_id: ID de la cámara
            confidence_mask: array (N,) booleano para filtrar puntos
        
        Returns:
            dict con estadísticas de error
        """
        reprojected_2d = self.reproject_points(points_3d, camera_id)
        
        if reprojected_2d is None:
            return {'mean_error': float('inf'), 'max_error': float('inf'), 'num_points': 0}
        
        # Aplicar máscara de confianza si se proporciona
        if confidence_mask is not None:
            valid_mask = confidence_mask & ~np.isnan(reprojected_2d).any(axis=1) & ~np.isnan(observed_2d).any(axis=1)
        else:
            valid_mask = ~np.isnan(reprojected_2d).any(axis=1) & ~np.isnan(observed_2d).any(axis=1)
        
        if not np.any(valid_mask):
            return {'mean_error': float('inf'), 'max_error': float('inf'), 'num_points': 0}
        
        # Calcular errores euclidanos
        valid_reprojected = reprojected_2d[valid_mask]
        valid_observed = observed_2d[valid_mask]
        
        errors = np.linalg.norm(valid_reprojected - valid_observed, axis=1)
        
        return {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'num_points': int(np.sum(valid_mask)),
            'errors': errors
        }
    
    def validate_frame_reconstruction(self, points_3d: np.ndarray,
                                    keypoints_2d: Dict[int, np.ndarray],
                                    confidences: Dict[int, np.ndarray],
                                    confidence_threshold: float = 0.3) -> Dict[int, Dict[str, float]]:
        """
        Validar reconstrucción de un frame completo
        
        Args:
            points_3d: array (N, 3) con puntos 3D reconstruidos
            keypoints_2d: Dict {camera_id: keypoints_array (N, 2)}
            confidences: Dict {camera_id: confidence_array (N,)}
            confidence_threshold: Umbral mínimo de confianza
        
        Returns:
            Dict {camera_id: estadísticas_error}
        """
        validation_results = {}
        
        for camera_id, observed_2d in keypoints_2d.items():
            if camera_id in confidences:
                confidence_mask = confidences[camera_id] > confidence_threshold
                
                error_stats = self.calculate_reprojection_errors(
                    points_3d, observed_2d, camera_id, confidence_mask
                )
                validation_results[camera_id] = error_stats
                
                logger.debug(f"Cámara {camera_id}: RMSE={error_stats['rmse']:.2f}px, "
                           f"puntos={error_stats['num_points']}")
        
        return validation_results
    
    def validate_sequence_reconstruction(self, points_3d_sequence: Dict[str, np.ndarray],
                                       keypoints_2d_sequence: Dict[int, Dict[str, np.ndarray]],
                                       confidences_sequence: Dict[int, Dict[str, np.ndarray]],
                                       confidence_threshold: float = 0.3) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Validar secuencia completa de reconstrucciones
        
        Returns:
            Dict {frame_key: {camera_id: estadísticas_error}}
        """
        sequence_results = {}
        
        for frame_key in points_3d_sequence.keys():
            # Recopilar keypoints 2D para este frame
            frame_keypoints_2d = {}
            frame_confidences = {}
            
            for camera_id in keypoints_2d_sequence.keys():
                if frame_key in keypoints_2d_sequence[camera_id]:
                    frame_keypoints_2d[camera_id] = keypoints_2d_sequence[camera_id][frame_key]
                    frame_confidences[camera_id] = confidences_sequence[camera_id][frame_key]
            
            if frame_keypoints_2d:
                frame_results = self.validate_frame_reconstruction(
                    points_3d_sequence[frame_key],
                    frame_keypoints_2d,
                    frame_confidences,
                    confidence_threshold
                )
                sequence_results[frame_key] = frame_results
        
        # Log estadísticas generales
        self._log_sequence_statistics(sequence_results)
        
        return sequence_results
    
    def _log_sequence_statistics(self, sequence_results: Dict[str, Dict[int, Dict[str, float]]]):
        """Log estadísticas generales de la secuencia"""
        all_errors = []
        camera_errors = {}
        
        for frame_key, frame_results in sequence_results.items():
            for camera_id, error_stats in frame_results.items():
                if 'errors' in error_stats:
                    all_errors.extend(error_stats['errors'])
                    
                    if camera_id not in camera_errors:
                        camera_errors[camera_id] = []
                    camera_errors[camera_id].extend(error_stats['errors'])
        
        if all_errors:
            overall_rmse = np.sqrt(np.mean(np.array(all_errors)**2))
            overall_mean = np.mean(all_errors)
            overall_max = np.max(all_errors)
            
            logger.info(f"Validación completa - RMSE global: {overall_rmse:.2f}px, "
                       f"Error medio: {overall_mean:.2f}px, Error máximo: {overall_max:.2f}px")
            
            # Estadísticas por cámara
            for camera_id, errors in camera_errors.items():
                camera_rmse = np.sqrt(np.mean(np.array(errors)**2))
                logger.info(f"Cámara {camera_id}: RMSE={camera_rmse:.2f}px, "
                           f"puntos={len(errors)}")
    
    def create_reprojection_visualization(self, points_3d: np.ndarray,
                                        keypoints_2d: Dict[int, np.ndarray],
                                        confidences: Dict[int, np.ndarray],
                                        save_path: Optional[Path] = None,
                                        confidence_threshold: float = 0.3) -> Optional[np.ndarray]:
        """
        Crear visualización de errores de reproyección
        
        Returns:
            Array de imagen de visualización o None si no es posible
        """
        try:
            import matplotlib.pyplot as plt
            
            num_cameras = len(keypoints_2d)
            fig, axes = plt.subplots(1, num_cameras, figsize=(5*num_cameras, 5))
            
            if num_cameras == 1:
                axes = [axes]
            
            for i, (camera_id, observed_2d) in enumerate(keypoints_2d.items()):
                reprojected_2d = self.reproject_points(points_3d, camera_id)
                
                if reprojected_2d is not None and camera_id in confidences:
                    confidence_mask = confidences[camera_id] > confidence_threshold
                    valid_mask = (confidence_mask & 
                                 ~np.isnan(reprojected_2d).any(axis=1) & 
                                 ~np.isnan(observed_2d).any(axis=1))
                    
                    if np.any(valid_mask):
                        valid_observed = observed_2d[valid_mask]
                        valid_reprojected = reprojected_2d[valid_mask]
                        
                        # Scatter plot de puntos observados vs reproyectados
                        axes[i].scatter(valid_observed[:, 0], valid_observed[:, 1], 
                                      c='blue', alpha=0.6, label='Observado', s=30)
                        axes[i].scatter(valid_reprojected[:, 0], valid_reprojected[:, 1], 
                                      c='red', alpha=0.6, label='Reproyectado', s=30)
                        
                        # Líneas de error
                        for j in range(len(valid_observed)):
                            axes[i].plot([valid_observed[j, 0], valid_reprojected[j, 0]],
                                       [valid_observed[j, 1], valid_reprojected[j, 1]], 
                                       'gray', alpha=0.3, linewidth=1)
                        
                        # Calcular estadísticas
                        errors = np.linalg.norm(valid_observed - valid_reprojected, axis=1)
                        rmse = np.sqrt(np.mean(errors**2))
                        
                        axes[i].set_title(f'Cámara {camera_id}\nRMSE: {rmse:.2f}px')
                        axes[i].set_xlabel('X (píxeles)')
                        axes[i].set_ylabel('Y (píxeles)')
                        axes[i].legend()
                        axes[i].grid(True, alpha=0.3)
                        axes[i].set_aspect('equal')
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Visualización guardada en {save_path}")
            
            # Convertir a array de imagen
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return img_array
            
        except ImportError:
            logger.warning("Matplotlib no disponible para visualización")
            return None
        except Exception as e:
            logger.error(f"Error creando visualización: {e}")
            return None
    
    def set_error_threshold(self, threshold: float):
        """Configurar umbral de error aceptable"""
        self.error_threshold = threshold
        logger.info(f"Umbral de error configurado: {threshold} píxeles")


def validate_reconstruction(camera_system: CameraSystem,
                          points_3d_sequence: Dict[str, np.ndarray],
                          keypoints_2d_sequence: Dict[int, Dict[str, np.ndarray]],
                          confidences_sequence: Dict[int, Dict[str, np.ndarray]],
                          confidence_threshold: float = 0.3,
                          save_visualizations: bool = False,
                          output_dir: Optional[Path] = None) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Función principal para validación de reconstrucción
    """
    validator = ReprojectionValidator(camera_system)
    
    # Validar secuencia completa
    results = validator.validate_sequence_reconstruction(
        points_3d_sequence, keypoints_2d_sequence, confidences_sequence, confidence_threshold
    )
    
    # Crear visualizaciones si se solicita
    if save_visualizations and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for frame_key in list(points_3d_sequence.keys())[:5]:  # Solo primeros 5 frames
            frame_keypoints_2d = {}
            frame_confidences = {}
            
            for camera_id in keypoints_2d_sequence.keys():
                if frame_key in keypoints_2d_sequence[camera_id]:
                    frame_keypoints_2d[camera_id] = keypoints_2d_sequence[camera_id][frame_key]
                    frame_confidences[camera_id] = confidences_sequence[camera_id][frame_key]
            
            if frame_keypoints_2d:
                viz_path = output_dir / f"reprojection_{frame_key}.png"
                validator.create_reprojection_visualization(
                    points_3d_sequence[frame_key],
                    frame_keypoints_2d,
                    frame_confidences,
                    viz_path,
                    confidence_threshold
                )
    
    return results

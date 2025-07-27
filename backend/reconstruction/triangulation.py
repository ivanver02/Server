"""
Sistema de triangulación 3D para reconstrucción de keypoints
Implementa múltiples métodos de triangulación con optimización
Actualizado para usar la nueva estructura de archivos separados
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TriangulationResult:
    """Resultado de triangulación 3D"""
    points_3d: np.ndarray  # Puntos 3D triangulados (N, 3)
    reprojection_errors: Dict[int, float]  # Error por cámara
    confidence_scores: np.ndarray  # Confianza de cada punto (N,)
    triangulation_method: str  # Método usado
    processing_info: Dict[str, Any]  # Info adicional

class Triangulator:
    """
    Sistema de triangulación 3D para múltiples cámaras
    Implementa diferentes métodos con optimización opcional
    """
    
    def __init__(self):
        self.cameras: Dict[int, Any] = {}  # {camera_id: Camera}
        self.max_reprojection_error = 5.0  # píxeles
        self.min_cameras_for_point = 2
        
    def set_cameras(self, cameras: Dict[int, Any]): # No se usa por ahora
        """
        Establecer cámaras para triangulación
        
        Args:
            cameras: Diccionario {camera_id: Camera}
        """
        self.cameras = cameras.copy()
        
        # Verificar que las cámaras estén calibradas
        calibrated_count = 0
        for camera_id, camera in self.cameras.items():
            if camera.intrinsics_calibrated and camera.extrinsics_calibrated:
                calibrated_count += 1
            else:
                logger.warning(f"Cámara {camera_id} no está completamente calibrada")
        
        logger.info(f"Triangulador configurado con {calibrated_count}/{len(self.cameras)} cámaras calibradas")

    def _compute_point_reprojection_error(self, point_3d: np.ndarray, 
                                        points_2d_observed: List[np.ndarray],
                                        projection_matrices: List[np.ndarray]) -> float:
        """
        Calcular error de reproyección para un punto 3D
        
        Args:
            point_3d: Punto 3D (3,)
            points_2d_observed: Puntos 2D observados
            projection_matrices: Matrices de proyección
            
        Returns:
            Error RMS de reproyección en píxeles
        """
        try:
            errors = []
            point_3d_homogeneous = np.append(point_3d, 1.0)  # Convertir a homogéneas
            
            for point_2d_obs, P in zip(points_2d_observed, projection_matrices):
                # Proyectar punto 3D
                point_2d_proj_homogeneous = P @ point_3d_homogeneous
                
                if abs(point_2d_proj_homogeneous[2]) > 1e-8:
                    point_2d_proj = point_2d_proj_homogeneous[:2] / point_2d_proj_homogeneous[2]
                    
                    # Calcular error euclidiano
                    error = np.linalg.norm(point_2d_proj - point_2d_obs)
                    errors.append(error)
            
            return np.sqrt(np.mean(np.array(errors)**2)) if errors else float('inf')
            
        except Exception as e:
            logger.debug(f"Error calculando reproyección: {e}")
            return float('inf')
    

    def _compute_camera_reprojection_errors(self, points_3d: np.ndarray,
                                          keypoints_dict: Dict[int, np.ndarray],
                                          cameras_dict: Dict[int, Any]) -> Dict[int, float]:
        """
        Calcular error de reproyección para cada cámara
        
        Args:
            points_3d: Puntos 3D triangulados (N, 3)
            keypoints_dict: {camera_id: keypoints_2d}
            cameras_dict: {camera_id: Camera}
            
        Returns:
            {camera_id: error_rms}
        """
        errors = {}
        
        try:
            for camera_id, camera in cameras_dict.items():
                if camera_id in keypoints_dict and len(points_3d) > 0:
                    keypoints_2d = keypoints_dict[camera_id]
                    n_points = min(len(points_3d), len(keypoints_2d))
                    
                    if n_points > 0:
                        # Proyectar puntos 3D a la cámara
                        projected_2d = camera.project_3d_to_2d(points_3d[:n_points])
                        
                        if len(projected_2d) > 0:
                            # Calcular error
                            observed_2d = keypoints_2d[:n_points]
                            differences = projected_2d - observed_2d
                            rms_error = np.sqrt(np.mean(np.sum(differences**2, axis=1)))
                            errors[camera_id] = rms_error
            
            return errors
            
        except Exception as e:
            logger.error(f"Error calculando errores de cámara: {e}")
            return {}

    def _triangulate_point_dlt(self, points_2d: List[np.ndarray], 
                              projection_matrices: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Triangular un punto usando Direct Linear Transform
        
        Args:
            points_2d: Lista de puntos 2D (cada uno shape (2,))
            projection_matrices: Lista de matrices de proyección (cada una 3x4)
            
        Returns:
            Punto 3D (3,) o None si falla
        """
        try:
            n_views = len(points_2d)
            if n_views < 2:
                return None
            
            # Construir sistema de ecuaciones Ax = 0
            A = np.zeros((2 * n_views, 4))
            
            for i, (point_2d, P) in enumerate(zip(points_2d, projection_matrices)):
                x, y = point_2d
                
                # Ecuaciones del DLT
                A[2*i] = x * P[2] - P[0]      # x * P[2,:] - P[0,:]
                A[2*i + 1] = y * P[2] - P[1]  # y * P[2,:] - P[1,:]
            
            # Resolver usando SVD
            _, _, Vt = np.linalg.svd(A)
            point_3d_homogeneous = Vt[-1]  # Última fila de V (última columna de V^T)
            
            # Convertir de coordenadas homogéneas a cartesianas
            if abs(point_3d_homogeneous[3]) > 1e-8:
                point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
                return point_3d
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error en DLT para punto: {e}")
            return None
    
    def triangulate_dlt(self, keypoints_multi_camera: Dict[int, np.ndarray]) -> TriangulationResult: # Primera opción para triangulación
        """
        Triangulación usando Direct Linear Transform (DLT)
        
        Args:
            keypoints_multi_camera: {camera_id: keypoints_2d}
            
        Returns:
            Resultado de triangulación
        """
        try:
            logger.debug(f"Triangulando con DLT: {len(keypoints_multi_camera)} cámaras")
            
            # Filtrar cámaras calibradas
            valid_cameras = {}
            valid_keypoints = {}
            
            for camera_id, keypoints in keypoints_multi_camera.items():
                if (camera_id in self.cameras and 
                    self.cameras[camera_id].projection_matrix is not None and 
                    len(keypoints) > 0):
                    valid_cameras[camera_id] = self.cameras[camera_id]
                    valid_keypoints[camera_id] = keypoints
            
            if len(valid_cameras) < self.min_cameras_for_point:
                return TriangulationResult(
                    points_3d=np.array([]),
                    reprojection_errors={},
                    confidence_scores=np.array([]),
                    triangulation_method='dlt',
                    processing_info={'error': 'Insuficientes cámaras válidas'}
                )
            
            # Determinar número de puntos a triangular
            n_points = min(len(kp) for kp in valid_keypoints.values())
            if n_points == 0:
                return TriangulationResult(
                    points_3d=np.array([]),
                    reprojection_errors={},
                    confidence_scores=np.array([]),
                    triangulation_method='dlt',
                    processing_info={'error': 'No hay keypoints para triangular'}
                )
            
            points_3d = []
            point_confidences = []
            
            # Triangular cada punto individualmente
            for point_idx in range(n_points):
                point_2d_list = []
                projection_matrices = []
                
                # Recopilar observaciones del punto en todas las cámaras
                for camera_id in valid_cameras:
                    if point_idx < len(valid_keypoints[camera_id]):
                        point_2d = valid_keypoints[camera_id][point_idx]
                        projection_matrix = valid_cameras[camera_id].projection_matrix
                        
                        point_2d_list.append(point_2d)
                        projection_matrices.append(projection_matrix)
                
                if len(point_2d_list) >= self.min_cameras_for_point:
                    # Triangular punto usando DLT
                    point_3d = self._triangulate_point_dlt(point_2d_list, projection_matrices)
                    
                    if point_3d is not None:
                        # Calcular confianza basada en error de reproyección
                        reprojection_error = self._compute_point_reprojection_error(
                            point_3d, point_2d_list, projection_matrices
                        )
                        
                        confidence = max(0.0, 1.0 - reprojection_error / self.max_reprojection_error)
                        
                        if reprojection_error <= self.max_reprojection_error:
                            points_3d.append(point_3d)
                            point_confidences.append(confidence)
                        else:
                            logger.debug(f"Punto {point_idx} descartado por alto error: {reprojection_error:.2f}")
            
            # Convertir a arrays numpy
            if points_3d:
                points_3d = np.array(points_3d)
                point_confidences = np.array(point_confidences)
            else:
                points_3d = np.array([]).reshape(0, 3)
                point_confidences = np.array([])
            
            # Calcular errores de reproyección por cámara
            reprojection_errors = self._compute_camera_reprojection_errors(
                points_3d, valid_keypoints, valid_cameras
            )
            
            result = TriangulationResult(
                points_3d=points_3d,
                reprojection_errors=reprojection_errors,
                confidence_scores=point_confidences,
                triangulation_method='dlt',
                processing_info={
                    'input_points': n_points,
                    'triangulated_points': len(points_3d),
                    'cameras_used': list(valid_cameras.keys()),
                    'mean_reprojection_error': np.mean(list(reprojection_errors.values())) if reprojection_errors else 0.0
                }
            )
            
            logger.debug(f"DLT completado: {len(points_3d)}/{n_points} puntos triangulados")
            return result
            
        except Exception as e:
            logger.error(f"Error en triangulación DLT: {e}")
            return TriangulationResult(
                points_3d=np.array([]),
                reprojection_errors={},
                confidence_scores=np.array([]),
                triangulation_method='dlt',
                processing_info={'error': str(e)}
            )
    
    def triangulate_opencv(self, keypoints_multi_camera: Dict[int, np.ndarray]) -> TriangulationResult: # Segunda opción para triangulación
        """
        Triangulación usando cv2.triangulatePoints (solo para 2 cámaras)
        
        Args:
            keypoints_multi_camera: {camera_id: keypoints_2d}
            
        Returns:
            Resultado de triangulación
        """
        try:
            # Filtrar cámaras válidas
            valid_data = []
            for camera_id, keypoints in keypoints_multi_camera.items():
                if (camera_id in self.cameras and 
                    self.cameras[camera_id].projection_matrix is not None and 
                    len(keypoints) > 0):
                    valid_data.append((camera_id, keypoints, self.cameras[camera_id]))
            
            if len(valid_data) < 2:
                return TriangulationResult(
                    points_3d=np.array([]),
                    reprojection_errors={},
                    confidence_scores=np.array([]),
                    triangulation_method='opencv',
                    processing_info={'error': 'Se necesitan al menos 2 cámaras válidas'}
                )
            
            # Para múltiples cámaras, usar la primera y segunda
            camera1_id, keypoints1, camera1 = valid_data[0]
            camera2_id, keypoints2, camera2 = valid_data[1]
            
            # Tomar puntos comunes
            n_points = min(len(keypoints1), len(keypoints2))
            if n_points == 0:
                return TriangulationResult(
                    points_3d=np.array([]),
                    reprojection_errors={},
                    confidence_scores=np.array([]),
                    triangulation_method='opencv',
                    processing_info={'error': 'No hay keypoints comunes'}
                )
            
            points1 = keypoints1[:n_points].T  # (2, N)
            points2 = keypoints2[:n_points].T  # (2, N)
            
            # Triangular usando OpenCV
            points_4d = cv2.triangulatePoints(
                camera1.projection_matrix,
                camera2.projection_matrix,
                points1.astype(np.float32),
                points2.astype(np.float32)
            )
            
            # Convertir de homogéneas a cartesianas
            points_3d = (points_4d[:3] / points_4d[3]).T  # (N, 3)
            
            # Calcular confianzas basadas en reproyección
            confidences = []
            for i, point_3d in enumerate(points_3d):
                error1 = np.linalg.norm(camera1.project_3d_to_2d(point_3d.reshape(1, -1))[0] - keypoints1[i])
                error2 = np.linalg.norm(camera2.project_3d_to_2d(point_3d.reshape(1, -1))[0] - keypoints2[i])
                avg_error = (error1 + error2) / 2
                confidence = max(0.0, 1.0 - avg_error / self.max_reprojection_error)
                confidences.append(confidence)
            
            confidences = np.array(confidences)
            
            # Calcular errores por cámara
            reprojection_errors = self._compute_camera_reprojection_errors(
                points_3d, {camera1_id: keypoints1, camera2_id: keypoints2}, 
                {camera1_id: camera1, camera2_id: camera2}
            )
            
            result = TriangulationResult(
                points_3d=points_3d,
                reprojection_errors=reprojection_errors,
                confidence_scores=confidences,
                triangulation_method='opencv',
                processing_info={
                    'input_points': n_points,
                    'triangulated_points': len(points_3d),
                    'cameras_used': [camera1_id, camera2_id],
                    'mean_reprojection_error': np.mean(list(reprojection_errors.values())) if reprojection_errors else 0.0
                }
            )
            
            logger.debug(f"OpenCV triangulación completada: {len(points_3d)} puntos")
            return result
            
        except Exception as e:
            logger.error(f"Error en triangulación OpenCV: {e}")
            return TriangulationResult(
                points_3d=np.array([]),
                reprojection_errors={},
                confidence_scores=np.array([]),
                triangulation_method='opencv',
                processing_info={'error': str(e)}
            )
        
    def _load_frame_keypoints(self, patient_id: str, session_id: str, 
                            global_frame: int, detector_name: str = 'VitPose') -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Cargar keypoints 2D y confianza de un frame usando nuevas funciones auxiliares
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión  
            global_frame: Número global del frame
            detector_name: Detector a usar para triangulación
            
        Returns:
            {camera_id: (keypoints_2d, confidence)}
        """
        keypoints_dict = {}
        
        try:
            # Importar funciones auxiliares
            from backend.processing.utils import load_keypoints_2d_frame
            from config import data_config
            
            session_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}"
            
            if not session_dir.exists():
                return keypoints_dict
            
            # Buscar en cada cámara
            for camera_dir in session_dir.iterdir():
                if camera_dir.is_dir() and camera_dir.name.startswith('camera'):
                    try:
                        camera_id = int(camera_dir.name.replace('camera', ''))
                        
                        # Cargar keypoints y confianza usando función auxiliar
                        keypoints, confidence = load_keypoints_2d_frame(
                            detector_name=detector_name,
                            camera_id=camera_id,
                            global_frame=global_frame,
                            patient_id=patient_id,
                            session_id=session_id
                        )
                        
                        if keypoints is not None and confidence is not None:
                            keypoints_dict[camera_id] = (keypoints, confidence)
                            
                    except ValueError:
                        continue
            
            logger.debug(f"Frame {global_frame}: keypoints de {len(keypoints_dict)} cámaras cargados")
            return keypoints_dict
            
        except Exception as e:
            logger.error(f"Error cargando keypoints frame {global_frame}: {e}")
            return {}
    
    
    def triangulate_frame(self, patient_id: str, session_id: str, 
                         global_frame: int, method: str = 'dlt') -> TriangulationResult: # No se usa por ahora
        """
        Triangular keypoints 2D de un frame específico
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            global_frame: Número global del frame
            method: Método de triangulación ('dlt' o 'opencv')
            
        Returns:
            Resultado de triangulación
        """
        try:
            # Cargar keypoints 2D de todas las cámaras (con confianza)
            keypoints_with_confidence = self._load_frame_keypoints(
                patient_id, session_id, global_frame
            )
            
            if not keypoints_with_confidence:
                return TriangulationResult(
                    points_3d=np.array([]),
                    reprojection_errors={},
                    confidence_scores=np.array([]),
                    triangulation_method=method,
                    processing_info={'error': f'No hay keypoints para frame {global_frame}'}
                )
            
            # Extraer solo keypoints para triangulación (manteniendo compatibilidad)
            keypoints_multi_camera = {}
            confidence_multi_camera = {}
            for camera_id, (keypoints, confidence) in keypoints_with_confidence.items():
                keypoints_multi_camera[camera_id] = keypoints
                confidence_multi_camera[camera_id] = confidence
            
            # Aplicar método de triangulación
            if method == 'dlt':
                result = self.triangulate_dlt(keypoints_multi_camera)
            elif method == 'opencv':
                result = self.triangulate_opencv(keypoints_multi_camera)
            else:
                raise ValueError(f"Método no soportado: {method}")
            
            # Incorporar información de confianza en el resultado
            if len(result.confidence_scores) == 0 and confidence_multi_camera:
                # Calcular confianza promedio por keypoint desde todas las cámaras
                avg_confidence = []
                num_keypoints = len(result.points_3d) if len(result.points_3d) > 0 else 0
                
                for kp_idx in range(num_keypoints):
                    confidences = []
                    for camera_id in confidence_multi_camera:
                        if kp_idx < len(confidence_multi_camera[camera_id]):
                            confidences.append(confidence_multi_camera[camera_id][kp_idx])
                    avg_confidence.append(np.mean(confidences) if confidences else 0.0)
                
                result.confidence_scores = np.array(avg_confidence)
            
            # Añadir información del frame
            result.processing_info.update({
                'patient_id': patient_id,
                'session_id': session_id,
                'global_frame': global_frame
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error triangulando frame {global_frame}: {e}")
            return TriangulationResult(
                points_3d=np.array([]),
                reprojection_errors={},
                confidence_scores=np.array([]),
                triangulation_method=method,
                processing_info={'error': str(e)}
            )
    
    def save_triangulation_result(self, result: TriangulationResult, 
                                patient_id: str, session_id: str, 
                                global_frame: int) -> bool: # No se usa por ahora
        """
        Guardar resultado de triangulación en disco
        
        Args:
            result: Resultado de triangulación
            patient_id: ID del paciente
            session_id: ID de la sesión
            global_frame: Número global del frame
            
        Returns:
            True si se guardó correctamente
        """
        try:
            from config import data_config
            
            # Directorio destino
            save_dir = data_config.keypoints_3d_dir / f"patient{patient_id}" / f"session{session_id}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar puntos 3D
            points_file = save_dir / f"{global_frame}.npy"
            np.save(points_file, result.points_3d)
            
            # Guardar confianzas
            confidence_file = save_dir / f"{global_frame}_confidence.npy"
            np.save(confidence_file, result.confidence_scores)
            
            # Guardar información adicional
            info_file = save_dir / f"{global_frame}_info.npz"
            np.savez(info_file, 
                    reprojection_errors=result.reprojection_errors,
                    triangulation_method=result.triangulation_method,
                    processing_info=result.processing_info)
            
            logger.debug(f"Triangulación guardada: {points_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando triangulación: {e}")
            return False

# Instancia global del triangulador
triangulator = Triangulator()

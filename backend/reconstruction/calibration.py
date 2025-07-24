"""
Sistema de calibración de cámaras para reconstrucción 3D
Maneja calibración intrínseca (tablero de ajedrez) y extrínseca (keypoints 2D)
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from scipy.optimize import least_squares
from itertools import combinations

from .camera import Camera, CameraCalibrationResult

logger = logging.getLogger(__name__)

class CalibrationSystem:
    """
    Sistema de calibración completo para múltiples cámaras
    """
    
    def __init__(self):
        self.cameras: Dict[int, Camera] = {}
        self.reference_camera_id: Optional[int] = None
        self.calibration_quality_threshold = 1.0  # Error máximo en píxeles
        
    def add_camera(self, camera_id: int, serial_number: str = None) -> Camera:
        """
        Añadir cámara al sistema
        
        Args:
            camera_id: ID de la cámara
            serial_number: Número de serie (opcional)
            
        Returns:
            Objeto Camera creado
        """
        camera = Camera(camera_id, serial_number)
        self.cameras[camera_id] = camera
        
        # La primera cámara se convierte en referencia por defecto
        if self.reference_camera_id is None:
            self.set_reference_camera(camera_id)
        
        logger.info(f"📷 Cámara {camera_id} añadida al sistema")
        return camera
    
    def set_reference_camera(self, camera_id: int):
        """
        Establecer cámara de referencia (origen del sistema de coordenadas)
        
        Args:
            camera_id: ID de la cámara de referencia
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Cámara {camera_id} no existe en el sistema")
        
        # Limpiar referencia anterior
        if self.reference_camera_id is not None:
            self.cameras[self.reference_camera_id].is_reference_camera = False
        
        # Establecer nueva referencia
        self.reference_camera_id = camera_id
        self.cameras[camera_id].set_as_reference()
        
        logger.info(f"📐 Cámara {camera_id} establecida como referencia")
    
    def calibrate_camera_intrinsics(self, camera_id: int, chessboard_images: List[np.ndarray]) -> CameraCalibrationResult:
        """
        Calibrar parámetros intrínsecos de una cámara usando tablero de ajedrez
        
        Args:
            camera_id: ID de la cámara
            chessboard_images: Lista de imágenes con tablero de ajedrez
            
        Returns:
            Resultado de la calibración
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Cámara {camera_id} no existe")
        
        from config import camera_intrinsics
        chessboard_size = camera_intrinsics.CHESSBOARD_SIZE
        square_size = camera_intrinsics.SQUARE_SIZE
        
        logger.info(f"🎯 Calibrando intrínsecos cámara {camera_id} con {len(chessboard_images)} imágenes")
        
        result = self.cameras[camera_id].calibrate_intrinsics_from_chessboard(
            chessboard_images, chessboard_size, square_size
        )
        
        if result.success:
            # Actualizar configuración global si la calibración es buena
            if result.reprojection_error < self.calibration_quality_threshold:
                from config import update_camera_intrinsics
                update_camera_intrinsics(
                    camera_id, 
                    result.camera_matrix, 
                    result.distortion_coeffs,
                    self.cameras[camera_id].serial_number
                )
                logger.info(f"✅ Intrínsecos actualizados globalmente para cámara {camera_id}")
            else:
                logger.warning(f"⚠️ Calibración de baja calidad para cámara {camera_id}: {result.reprojection_error:.3f}")
        
        return result
    
    def estimate_camera_pose_from_keypoints(self, camera_id: int, keypoints_2d_multi_frame: List[np.ndarray],
                                          reference_3d_points: np.ndarray) -> bool:
        """
        Estimar pose de cámara usando keypoints 2D y puntos 3D de referencia
        
        Args:
            camera_id: ID de la cámara
            keypoints_2d_multi_frame: Lista de keypoints 2D para múltiples frames
            reference_3d_points: Puntos 3D de referencia
            
        Returns:
            True si la estimación fue exitosa
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Cámara {camera_id} no existe")
        
        camera = self.cameras[camera_id]
        
        if not camera.intrinsics_calibrated:
            logger.error(f"Cámara {camera_id} no tiene intrínsecos calibrados")
            return False
        
        try:
            logger.info(f"🎯 Estimando pose para cámara {camera_id} usando {len(keypoints_2d_multi_frame)} frames")
            
            # Usar múltiples frames para estimación robusta
            best_error = float('inf')
            best_rvec = None
            best_tvec = None
            
            for frame_keypoints in keypoints_2d_multi_frame:
                if len(frame_keypoints) < 4:  # Mínimo 4 puntos para solvePnP
                    continue
                
                # Tomar solo los puntos que tenemos en referencia 3D
                n_points = min(len(frame_keypoints), len(reference_3d_points))
                if n_points < 4:
                    continue
                
                object_points = reference_3d_points[:n_points]
                image_points = frame_keypoints[:n_points]
                
                # Resolver PnP
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera.camera_matrix,
                    camera.distortion_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Calcular error de reproyección
                    projected_points, _ = cv2.projectPoints(
                        object_points, rvec, tvec, 
                        camera.camera_matrix, camera.distortion_coeffs
                    )
                    
                    error = np.mean(np.sqrt(np.sum((projected_points.reshape(-1, 2) - image_points)**2, axis=1)))
                    
                    if error < best_error:
                        best_error = error
                        best_rvec = rvec
                        best_tvec = tvec
            
            if best_rvec is not None:
                # Convertir vector de rotación a matriz
                rotation_matrix, _ = cv2.Rodrigues(best_rvec)
                
                # Establecer parámetros extrínsecos
                camera.set_extrinsics(rotation_matrix, best_tvec)
                
                logger.info(f"✅ Pose estimada para cámara {camera_id}: error {best_error:.3f} píxeles")
                return True
            else:
                logger.error(f"❌ No se pudo estimar pose para cámara {camera_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error estimando pose para cámara {camera_id}: {e}")
            return False
    
    def calibrate_stereo_pair(self, camera1_id: int, camera2_id: int,
                             keypoints_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """
        Calibrar par estéreo usando correspondencias de keypoints
        
        Args:
            camera1_id: ID de la primera cámara
            camera2_id: ID de la segunda cámara  
            keypoints_pairs: Lista de pares (keypoints_cam1, keypoints_cam2)
            
        Returns:
            True si la calibración fue exitosa
        """
        if camera1_id not in self.cameras or camera2_id not in self.cameras:
            raise ValueError("Una o ambas cámaras no existen")
        
        camera1 = self.cameras[camera1_id]
        camera2 = self.cameras[camera2_id]
        
        if not (camera1.intrinsics_calibrated and camera2.intrinsics_calibrated):
            logger.error("Ambas cámaras deben tener intrínsecos calibrados")
            return False
        
        try:
            logger.info(f"🔗 Calibrando par estéreo: cámaras {camera1_id} - {camera2_id}")
            
            # Preparar puntos para calibración estéreo
            image_points1 = []
            image_points2 = []
            
            for kp1, kp2 in keypoints_pairs:
                if len(kp1) == len(kp2) and len(kp1) > 0:
                    image_points1.append(kp1)
                    image_points2.append(kp2)
            
            if len(image_points1) < 2:
                logger.error("Insuficientes correspondencias para calibración estéreo")
                return False
            
            # Usar calibración estéreo para obtener transformación relativa
            # Necesitamos puntos objeto, pero como no los tenemos, usamos estimación directa
            
            # Método alternativo: usar Essential Matrix
            # Concatenar todos los puntos
            all_points1 = np.vstack(image_points1)
            all_points2 = np.vstack(image_points2)
            
            # Calcular matriz esencial
            E, mask = cv2.findEssentialMat(
                all_points1, all_points2,
                camera1.camera_matrix, camera2.camera_matrix,
                cv2.RANSAC, 0.999, 1.0
            )
            
            if E is not None:
                # Recuperar pose de la matriz esencial
                _, R, t, _ = cv2.recoverPose(
                    E, all_points1, all_points2,
                    camera1.camera_matrix, camera2.camera_matrix
                )
                
                # Si camera1 es la referencia, establecer extrínsecos de camera2
                if camera1_id == self.reference_camera_id:
                    camera2.set_extrinsics(R, t)
                elif camera2_id == self.reference_camera_id:
                    # Invertir transformación
                    R_inv = R.T
                    t_inv = -R_inv @ t
                    camera1.set_extrinsics(R_inv, t_inv)
                else:
                    # Ninguna es referencia, establecer camera1 como tal
                    self.set_reference_camera(camera1_id)
                    camera2.set_extrinsics(R, t)
                
                logger.info(f"✅ Calibración estéreo exitosa para cámaras {camera1_id} - {camera2_id}")
                return True
            else:
                logger.error("❌ No se pudo calcular matriz esencial")
                return False
                
        except Exception as e:
            logger.error(f"Error en calibración estéreo: {e}")
            return False
    
    def auto_calibrate_extrinsics_from_session(self, patient_id: str, session_id: str) -> Dict[str, Any]:
        """
        Calibrar automáticamente parámetros extrínsecos usando keypoints de una sesión
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            
        Returns:
            Resultado de la calibración
        """
        try:
            logger.info(f"🔧 Auto-calibrando extrínsecos para sesión {patient_id}/{session_id}")
            
            from config import data_config
            
            # Buscar datos de keypoints 2D procesados
            session_dir = data_config.keypoints_2d_dir / f"patient{patient_id}" / f"session{session_id}"
            
            if not session_dir.exists():
                return {'error': 'Sesión no encontrada o sin procesar'}
            
            # Recopilar keypoints de todas las cámaras
            camera_keypoints = {}  # {camera_id: [frame_keypoints, ...]}
            
            for camera_dir in session_dir.iterdir():
                if camera_dir.is_dir() and camera_dir.name.startswith('camera'):
                    camera_id = int(camera_dir.name.replace('camera', ''))
                    
                    if camera_id not in self.cameras:
                        self.add_camera(camera_id)
                    
                    keypoints_files = sorted(camera_dir.glob("*.npy"), key=lambda x: int(x.stem))
                    frame_keypoints = []
                    
                    for kp_file in keypoints_files:
                        if '_confidence' not in kp_file.name:  # Solo archivos de keypoints, no confianza
                            keypoints = np.load(kp_file)
                            if len(keypoints) > 0:
                                frame_keypoints.append(keypoints)
                    
                    if frame_keypoints:
                        camera_keypoints[camera_id] = frame_keypoints
            
            if len(camera_keypoints) < 2:
                return {'error': 'Se necesitan al menos 2 cámaras con keypoints'}
            
            # Calibrar pares de cámaras
            calibrated_cameras = set()
            calibration_results = {}
            
            # Asegurar que tenemos una cámara de referencia
            if self.reference_camera_id is None:
                self.set_reference_camera(min(camera_keypoints.keys()))
            
            calibrated_cameras.add(self.reference_camera_id)
            
            # Calibrar el resto de cámaras con respecto a la referencia
            ref_keypoints = camera_keypoints[self.reference_camera_id]
            
            for camera_id in camera_keypoints:
                if camera_id == self.reference_camera_id:
                    continue
                
                # Preparar pares de correspondencias
                cam_keypoints = camera_keypoints[camera_id]
                n_frames = min(len(ref_keypoints), len(cam_keypoints))
                
                keypoints_pairs = []
                for i in range(n_frames):
                    ref_kp = ref_keypoints[i]
                    cam_kp = cam_keypoints[i]
                    
                    # Tomar solo los keypoints comunes (mismo número)
                    n_common = min(len(ref_kp), len(cam_kp))
                    if n_common >= 4:  # Mínimo para calibración
                        keypoints_pairs.append((ref_kp[:n_common], cam_kp[:n_common]))
                
                if len(keypoints_pairs) >= 2:
                    success = self.calibrate_stereo_pair(
                        self.reference_camera_id, camera_id, keypoints_pairs
                    )
                    
                    if success:
                        calibrated_cameras.add(camera_id)
                        calibration_results[camera_id] = 'success'
                    else:
                        calibration_results[camera_id] = 'failed'
                else:
                    calibration_results[camera_id] = 'insufficient_data'
            
            # Resultado final
            result = {
                'success': len(calibrated_cameras) >= 2,
                'reference_camera': self.reference_camera_id,
                'calibrated_cameras': list(calibrated_cameras),
                'calibration_results': calibration_results,
                'total_cameras': len(camera_keypoints),
                'calibrated_count': len(calibrated_cameras)
            }
            
            if result['success']:
                logger.info(f"✅ Auto-calibración exitosa: {len(calibrated_cameras)} cámaras calibradas")
            else:
                logger.error("❌ Auto-calibración falló")
            
            return result
            
        except Exception as e:
            error_msg = f"Error en auto-calibración: {e}"
            logger.error(error_msg)
            return {'error': error_msg}
    
    def get_all_cameras(self) -> Dict[int, Camera]:
        """Obtener todas las cámaras del sistema"""
        return self.cameras.copy()
    
    def get_camera(self, camera_id: int) -> Optional[Camera]:
        """Obtener cámara específica"""
        return self.cameras.get(camera_id)
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Obtener estado de calibración del sistema"""
        cameras_status = {}
        
        for camera_id, camera in self.cameras.items():
            cameras_status[camera_id] = camera.get_summary()
        
        return {
            'total_cameras': len(self.cameras),
            'reference_camera': self.reference_camera_id,
            'cameras': cameras_status,
            'fully_calibrated_cameras': sum(
                1 for cam in self.cameras.values() 
                if cam.intrinsics_calibrated and cam.extrinsics_calibrated
            )
        }
    
    def save_calibration(self, file_path: str):
        """Guardar calibración completa del sistema"""
        try:
            calibration_data = {
                'reference_camera_id': self.reference_camera_id,
                'cameras': {}
            }
            
            for camera_id, camera in self.cameras.items():
                calibration_data['cameras'][camera_id] = {
                    'camera_matrix': camera.camera_matrix,
                    'distortion_coeffs': camera.distortion_coeffs,
                    'rotation_matrix': camera.rotation_matrix,
                    'translation_vector': camera.translation_vector,
                    'is_reference_camera': camera.is_reference_camera,
                    'serial_number': camera.serial_number
                }
            
            np.savez(file_path, **calibration_data)
            logger.info(f"💾 Calibración del sistema guardada: {file_path}")
            
        except Exception as e:
            logger.error(f"Error guardando calibración: {e}")
    
    def load_calibration(self, file_path: str) -> bool:
        """
        Cargar calibración completa del sistema
        
        Returns:
            True si se cargó correctamente
        """
        try:
            data = np.load(file_path, allow_pickle=True).item()
            
            self.reference_camera_id = data.get('reference_camera_id')
            
            for camera_id, cam_data in data['cameras'].items():
                camera_id = int(camera_id)
                
                if camera_id not in self.cameras:
                    self.add_camera(camera_id, cam_data.get('serial_number'))
                
                camera = self.cameras[camera_id]
                camera.camera_matrix = cam_data['camera_matrix']
                camera.distortion_coeffs = cam_data['distortion_coeffs']
                
                if cam_data['rotation_matrix'] is not None:
                    camera.rotation_matrix = cam_data['rotation_matrix']
                    camera.translation_vector = cam_data['translation_vector']
                    camera.extrinsics_calibrated = True
                
                camera.is_reference_camera = cam_data.get('is_reference_camera', False)
                camera.intrinsics_calibrated = True
                camera._update_projection_matrix()
            
            logger.info(f"📂 Calibración del sistema cargada: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando calibración: {e}")
            return False

# Instancia global del sistema de calibración
calibration_system = CalibrationSystem()

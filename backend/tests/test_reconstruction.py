"""
Test de sistema completo de reconstrucción 3D
Valida el pipeline completo usando datos sintéticos y reales
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import json

# Imports del proyecto
from ..reconstruction.camera import Camera
from ..reconstruction.calibration import CalibrationSystem
from ..reconstruction.triangulation import Triangulator

logger = logging.getLogger(__name__)

class ReconstructionTester:
    """
    Sistema de testing para validar la reconstrucción 3D
    """
    
    def __init__(self):
        self.test_cameras = {}
        self.test_points_3d = None
        self.test_keypoints_2d = {}
        self.calibration_system = CalibrationSystem()
        self.triangulator = Triangulator()
        
    def generate_synthetic_data(self, n_cameras: int = 3, n_points: int = 17) -> Dict[str, Any]:
        """
        Generar datos sintéticos para testing
        
        Args:
            n_cameras: Número de cámaras
            n_points: Número de puntos 3D
            
        Returns:
            Datos generados
        """
        try:
            logger.info(f"🎲 Generando datos sintéticos: {n_cameras} cámaras, {n_points} puntos")
            
            # 1. Generar puntos 3D sintéticos (esqueleto humano aproximado)
            points_3d = self._generate_skeleton_points(n_points)
            self.test_points_3d = points_3d
            
            # 2. Generar cámaras sintéticas
            cameras = {}
            keypoints_2d = {}
            
            for camera_id in range(n_cameras):
                # Crear cámara con parámetros sintéticos
                camera = self._create_synthetic_camera(camera_id, n_cameras)
                cameras[camera_id] = camera
                
                # Proyectar puntos 3D a 2D
                projected_2d = camera.project_3d_to_2d(points_3d)
                
                # Añadir ruido realista
                noise_std = 1.0  # píxeles
                noisy_2d = projected_2d + np.random.normal(0, noise_std, projected_2d.shape)
                
                keypoints_2d[camera_id] = noisy_2d
            
            self.test_cameras = cameras
            self.test_keypoints_2d = keypoints_2d
            
            # Configurar sistemas
            self.calibration_system.cameras = cameras
            self.calibration_system.reference_camera_id = 0
            self.triangulator.set_cameras(cameras)
            
            synthetic_data = {
                'points_3d_ground_truth': points_3d,
                'cameras': cameras,
                'keypoints_2d': keypoints_2d,
                'n_cameras': n_cameras,
                'n_points': n_points
            }
            
            logger.info(f"✅ Datos sintéticos generados exitosamente")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generando datos sintéticos: {e}")
            return {}
    
    def _generate_skeleton_points(self, n_points: int) -> np.ndarray:
        """Generar puntos 3D que simulan un esqueleto humano"""
        if n_points == 17:  # COCO skeleton
            # Coordenadas aproximadas de esqueleto humano (en metros)
            points = np.array([
                [0.0, 0.0, 1.7],      # nose (0)
                [-0.05, 0.02, 1.72],  # left_eye (1)
                [0.05, 0.02, 1.72],   # right_eye (2)  
                [-0.08, 0.0, 1.68],   # left_ear (3)
                [0.08, 0.0, 1.68],    # right_ear (4)
                [-0.2, 0.0, 1.4],     # left_shoulder (5)
                [0.2, 0.0, 1.4],      # right_shoulder (6)
                [-0.35, 0.0, 1.15],   # left_elbow (7)
                [0.35, 0.0, 1.15],    # right_elbow (8)
                [-0.45, 0.0, 0.9],    # left_wrist (9)
                [0.45, 0.0, 0.9],     # right_wrist (10)
                [-0.15, 0.0, 0.9],    # left_hip (11)
                [0.15, 0.0, 0.9],     # right_hip (12)
                [-0.18, 0.0, 0.45],   # left_knee (13)
                [0.18, 0.0, 0.45],    # right_knee (14)
                [-0.15, 0.0, 0.0],    # left_ankle (15)
                [0.15, 0.0, 0.0]      # right_ankle (16)
            ])
        else:
            # Generar puntos aleatorios en un volumen razonable
            points = np.random.uniform(-1, 1, (n_points, 3))
            points[:, 2] += 1.0  # Elevar en Z para que esté por encima del suelo
        
        return points
    
    def _create_synthetic_camera(self, camera_id: int, total_cameras: int) -> Camera:
        """Crear cámara sintética con parámetros realistas"""
        camera = Camera(camera_id, f"SYNTHETIC_{camera_id}")
        
        # Parámetros intrínsecos realistas para Orbbec
        fx = fy = 640.0  # Focal length en píxeles
        cx, cy = 320.0, 240.0  # Centro óptico
        
        camera.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        camera.distortion_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)
        camera.intrinsics_calibrated = True
        
        # Parámetros extrínsecos: disponer cámaras en semicírculo
        if camera_id == 0:
            # Cámara de referencia
            camera.set_as_reference()
        else:
            # Distribuir otras cámaras alrededor
            angle = (camera_id * 60) * np.pi / 180  # Ángulos de 60 grados
            radius = 2.0  # 2 metros de distancia
            
            # Posición de la cámara
            x = radius * np.sin(angle)
            y = radius * np.cos(angle)
            z = 1.2  # Altura de 1.2m
            
            # Vector de traslación
            translation = np.array([[x], [y], [z]])
            
            # Matriz de rotación para apuntar hacia el centro
            # Crear rotación para que la cámara mire hacia (0, 0, 1)
            forward = np.array([0, 0, 1]) - np.array([x, y, z])
            forward = forward / np.linalg.norm(forward)
            
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            rotation = np.column_stack([right, up, -forward])
            
            camera.set_extrinsics(rotation, translation)
        
        return camera
    
    def test_triangulation_accuracy(self, method: str = 'dlt') -> Dict[str, Any]:
        """
        Test de precisión de triangulación
        
        Args:
            method: Método de triangulación ('dlt' o 'opencv')
            
        Returns:
            Resultados del test
        """
        try:
            logger.info(f"🎯 Testing triangulación con método: {method}")
            
            if not self.test_cameras or self.test_points_3d is None:
                logger.error("No hay datos sintéticos. Ejecuta generate_synthetic_data() primero")
                return {'error': 'No synthetic data available'}
            
            # Triangular usando el método especificado
            if method == 'dlt':
                result = self.triangulator.triangulate_dlt(self.test_keypoints_2d)
            elif method == 'opencv':
                result = self.triangulator.triangulate_opencv(self.test_keypoints_2d)
            else:
                return {'error': f'Método no soportado: {method}'}
            
            if len(result.points_3d) == 0:
                return {'error': 'No se triangularon puntos'}
            
            # Calcular errores con respecto al ground truth
            ground_truth = self.test_points_3d[:len(result.points_3d)]
            triangulated = result.points_3d
            
            # Error euclidiano por punto
            point_errors = np.linalg.norm(triangulated - ground_truth, axis=1)
            
            # Estadísticas de error
            results = {
                'method': method,
                'triangulated_points': len(result.points_3d),
                'ground_truth_points': len(self.test_points_3d),
                'mean_error_3d': float(np.mean(point_errors)),
                'std_error_3d': float(np.std(point_errors)),
                'max_error_3d': float(np.max(point_errors)),
                'min_error_3d': float(np.min(point_errors)),
                'rmse_3d': float(np.sqrt(np.mean(point_errors**2))),
                'reprojection_errors': result.reprojection_errors,
                'mean_reprojection_error': result.processing_info.get('mean_reprojection_error', 0),
                'confidence_scores': result.confidence_scores.tolist(),
                'mean_confidence': float(np.mean(result.confidence_scores)) if len(result.confidence_scores) > 0 else 0
            }
            
            logger.info(f"✅ Test triangulación completado - RMSE: {results['rmse_3d']:.4f}m, "
                       f"Error medio: {results['mean_error_3d']:.4f}m")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en test de triangulación: {e}")
            return {'error': str(e)}
    
    def test_reprojection_pipeline(self) -> Dict[str, Any]:
        """
        Test completo de pipeline: 3D → 2D → 3D
        Verifica la consistencia del sistema completo
        """
        try:
            logger.info("🔄 Testing pipeline completo de reproyección")
            
            if not self.test_cameras or self.test_points_3d is None:
                return {'error': 'No synthetic data available'}
            
            # 1. Proyectar puntos 3D ground truth a 2D
            projected_keypoints = {}
            for camera_id, camera in self.test_cameras.items():
                projected_2d = camera.project_3d_to_2d(self.test_points_3d)
                projected_keypoints[camera_id] = projected_2d
            
            # 2. Triangular de vuelta a 3D
            result = self.triangulator.triangulate_dlt(projected_keypoints)
            
            if len(result.points_3d) == 0:
                return {'error': 'Triangulación falló en pipeline test'}
            
            # 3. Calcular error de round-trip
            n_points = min(len(self.test_points_3d), len(result.points_3d))
            roundtrip_errors = np.linalg.norm(
                result.points_3d[:n_points] - self.test_points_3d[:n_points], 
                axis=1
            )
            
            # 4. Test de reproyección para cada cámara
            camera_reprojection_tests = {}
            for camera_id, camera in self.test_cameras.items():
                # Proyectar puntos triangulados
                reprojected_2d = camera.project_3d_to_2d(result.points_3d)
                original_2d = projected_keypoints[camera_id][:len(reprojected_2d)]
                
                # Error de reproyección
                reproj_errors = np.linalg.norm(reprojected_2d - original_2d, axis=1)
                
                camera_reprojection_tests[camera_id] = {
                    'mean_reprojection_error': float(np.mean(reproj_errors)),
                    'max_reprojection_error': float(np.max(reproj_errors)),
                    'rmse_reprojection': float(np.sqrt(np.mean(reproj_errors**2)))
                }
            
            results = {
                'pipeline_test': 'complete',
                'roundtrip_rmse': float(np.sqrt(np.mean(roundtrip_errors**2))),
                'roundtrip_mean_error': float(np.mean(roundtrip_errors)),
                'roundtrip_max_error': float(np.max(roundtrip_errors)),
                'points_processed': n_points,
                'camera_reprojection_tests': camera_reprojection_tests,
                'overall_reprojection_rmse': float(np.mean([
                    test['rmse_reprojection'] for test in camera_reprojection_tests.values()
                ]))
            }
            
            logger.info(f"✅ Pipeline test completado - Round-trip RMSE: {results['roundtrip_rmse']:.6f}m")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en test de pipeline: {e}")
            return {'error': str(e)}
    
    def test_real_data_session(self, patient_id: str, session_id: str) -> Dict[str, Any]:
        """
        Test usando datos reales de una sesión
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            
        Returns:
            Resultados del test
        """
        try:
            logger.info(f"🔍 Testing con datos reales: paciente {patient_id}, sesión {session_id}")
            
            from config import data_config
            
            # Verificar que existen keypoints 2D procesados
            session_dir = data_config.keypoints_2d_dir / f"patient{patient_id}" / f"session{session_id}"
            
            if not session_dir.exists():
                return {'error': f'Sesión no encontrada: {session_dir}'}
            
            # Auto-calibrar cámaras si es necesario
            if not self.calibration_system.cameras:
                calibration_result = self.calibration_system.auto_calibrate_extrinsics_from_session(
                    patient_id, session_id
                )
                
                if not calibration_result.get('success', False):
                    return {'error': f'Auto-calibración falló: {calibration_result}'}
                
                # Configurar triangulador
                self.triangulator.set_cameras(self.calibration_system.get_all_cameras())
            
            # Encontrar frames disponibles
            camera_dirs = [d for d in session_dir.iterdir() if d.is_dir() and d.name.startswith('camera')]
            
            if not camera_dirs:
                return {'error': 'No se encontraron datos de cámaras'}
            
            # Obtener frames comunes entre todas las cámaras
            common_frames = None
            for camera_dir in camera_dirs:
                frame_files = [f.stem for f in camera_dir.glob("*.npy") if '_confidence' not in f.name]
                frame_numbers = set(int(f) for f in frame_files)
                
                if common_frames is None:
                    common_frames = frame_numbers
                else:
                    common_frames = common_frames.intersection(frame_numbers)
            
            if not common_frames:
                return {'error': 'No hay frames comunes entre cámaras'}
            
            # Triangular una muestra de frames
            sample_frames = sorted(list(common_frames))[:10]  # Máximo 10 frames
            
            triangulation_results = []
            for frame_num in sample_frames:
                result = self.triangulator.triangulate_frame(
                    patient_id, session_id, frame_num, method='dlt'
                )
                
                if result.points_3d is not None and len(result.points_3d) > 0:
                    triangulation_results.append({
                        'frame': frame_num,
                        'points_3d_count': len(result.points_3d),
                        'mean_confidence': float(np.mean(result.confidence_scores)) if len(result.confidence_scores) > 0 else 0,
                        'reprojection_errors': result.reprojection_errors,
                        'mean_reprojection_error': result.processing_info.get('mean_reprojection_error', 0)
                    })
            
            if not triangulation_results:
                return {'error': 'No se pudo triangular ningún frame'}
            
            # Estadísticas generales
            results = {
                'test_type': 'real_data',
                'patient_id': patient_id,
                'session_id': session_id,
                'total_frames_available': len(common_frames),
                'frames_tested': len(triangulation_results),
                'average_points_per_frame': float(np.mean([r['points_3d_count'] for r in triangulation_results])),
                'average_confidence': float(np.mean([r['mean_confidence'] for r in triangulation_results])),
                'average_reprojection_error': float(np.mean([r['mean_reprojection_error'] for r in triangulation_results])),
                'calibration_status': self.calibration_system.get_calibration_status(),
                'frame_results': triangulation_results
            }
            
            logger.info(f"✅ Test datos reales completado - {len(triangulation_results)} frames triangulados")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en test con datos reales: {e}")
            return {'error': str(e)}
    
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """
        Ejecutar suite completo de tests
        
        Returns:
            Resultados de todos los tests
        """
        logger.info("🚀 Ejecutando suite completo de tests de reconstrucción 3D")
        
        test_results = {
            'timestamp': str(np.datetime64('now')),
            'test_suite': 'reconstruction_3d_complete'
        }
        
        try:
            # 1. Test con datos sintéticos
            logger.info("--- Test 1: Datos sintéticos ---")
            synthetic_data = self.generate_synthetic_data(n_cameras=3, n_points=17)
            
            if synthetic_data:
                test_results['synthetic_data_generation'] = {'success': True}
                
                # Test triangulación DLT
                dlt_results = self.test_triangulation_accuracy('dlt')
                test_results['triangulation_dlt'] = dlt_results
                
                # Test triangulación OpenCV
                opencv_results = self.test_triangulation_accuracy('opencv')
                test_results['triangulation_opencv'] = opencv_results
                
                # Test pipeline completo
                pipeline_results = self.test_reprojection_pipeline()
                test_results['reprojection_pipeline'] = pipeline_results
            else:
                test_results['synthetic_data_generation'] = {'success': False, 'error': 'Failed to generate data'}
            
            # 2. Resumen de resultados
            test_results['summary'] = self._generate_test_summary(test_results)
            
            logger.info("✅ Suite de tests completado")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error en suite de tests: {e}")
            test_results['error'] = str(e)
            return test_results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generar resumen de resultados de tests"""
        summary = {
            'tests_passed': 0,
            'tests_failed': 0,
            'overall_status': 'unknown'
        }
        
        try:
            # Verificar cada test
            if test_results.get('synthetic_data_generation', {}).get('success', False):
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1
            
            # DLT triangulation
            dlt_results = test_results.get('triangulation_dlt', {})
            if 'error' not in dlt_results and dlt_results.get('rmse_3d', float('inf')) < 0.01:  # < 1cm
                summary['tests_passed'] += 1
                summary['dlt_quality'] = 'excellent'
            elif 'error' not in dlt_results and dlt_results.get('rmse_3d', float('inf')) < 0.05:  # < 5cm
                summary['tests_passed'] += 1
                summary['dlt_quality'] = 'good'
            else:
                summary['tests_failed'] += 1
                summary['dlt_quality'] = 'poor'
            
            # Pipeline test
            pipeline_results = test_results.get('reprojection_pipeline', {})
            if 'error' not in pipeline_results and pipeline_results.get('roundtrip_rmse', float('inf')) < 1e-6:
                summary['tests_passed'] += 1
                summary['pipeline_quality'] = 'excellent'
            else:
                summary['tests_failed'] += 1
                summary['pipeline_quality'] = 'poor'
            
            # Estado general
            if summary['tests_failed'] == 0:
                summary['overall_status'] = 'all_passed'
            elif summary['tests_passed'] > summary['tests_failed']:
                summary['overall_status'] = 'mostly_passed'
            else:
                summary['overall_status'] = 'mostly_failed'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return summary
    
    def save_test_results(self, test_results: Dict[str, Any], output_file: str):
        """Guardar resultados de tests en archivo JSON"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            logger.info(f"💾 Resultados guardados en: {output_path}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")

# Función de utilidad para ejecutar tests rápidamente
def run_quick_test() -> Dict[str, Any]:
    """Ejecutar test rápido del sistema de reconstrucción"""
    tester = ReconstructionTester()
    return tester.run_complete_test_suite()

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar tests
    tester = ReconstructionTester()
    results = tester.run_complete_test_suite()
    
    # Guardar resultados
    from config import data_config
    output_file = data_config.logs_dir / "reconstruction_test_results.json"
    tester.save_test_results(results, str(output_file))
    
    # Mostrar resumen
    summary = results.get('summary', {})
    print(f"\n🎯 RESUMEN DE TESTS:")
    print(f"Tests pasados: {summary.get('tests_passed', 0)}")
    print(f"Tests fallidos: {summary.get('tests_failed', 0)}")
    print(f"Estado general: {summary.get('overall_status', 'unknown')}")
    
    if 'triangulation_dlt' in results:
        dlt_rmse = results['triangulation_dlt'].get('rmse_3d', 0)
        print(f"Precisión DLT: {dlt_rmse:.6f}m RMSE")
    
    print(f"\nResultados completos guardados en: {output_file}")

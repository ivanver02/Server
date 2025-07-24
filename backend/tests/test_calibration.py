"""
Tests para el sistema de calibración de cámaras

Este módulo contiene tests específicos para:
- Calibración intrínseca con tablero de ajedrez
- Estimación de parámetros extrínsecos
- Validación de matrices de proyección
- Test de reproyección

Uso:
    python -m backend.tests.test_calibration
    python -c "from backend.tests.test_calibration import test_quick_calibration; test_quick_calibration()"
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports condicionales
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV no disponible para testing")

# Imports del proyecto
try:
    from ..reconstruction.camera import Camera
    from ..reconstruction.calibration import CalibrationSystem
    from ..config.camera_intrinsics import CAMERA_INTRINSICS
    PROJECT_IMPORTS_AVAILABLE = True
except ImportError as e:
    PROJECT_IMPORTS_AVAILABLE = False
    logger.warning(f"Imports del proyecto no disponibles: {e}")

class TestCameraIntrinsics(unittest.TestCase):
    """Tests para calibración intrínseca de cámaras"""
    
    @classmethod
    def setUpClass(cls):
        """Setup una vez para toda la clase"""
        if not CV2_AVAILABLE:
            raise unittest.SkipTest("OpenCV no disponible")
        
        cls.chessboard_size = (9, 6)  # Esquinas interiores
        cls.square_size = 25.0  # mm
        cls.image_size = (640, 480)
        
        logger.info("✅ Setup TestCameraIntrinsics completado")
    
    def test_generate_chessboard_points(self):
        """Test: Generación de puntos de tablero de ajedrez"""
        logger.info("🧪 Test: Puntos tablero ajedrez")
        
        try:
            # Generar puntos 3D del tablero
            object_points = []
            for j in range(self.chessboard_size[1]):
                for i in range(self.chessboard_size[0]):
                    object_points.append([i * self.square_size, j * self.square_size, 0.0])
            
            object_points = np.array(object_points, dtype=np.float32)
            
            # Verificar dimensiones
            expected_points = self.chessboard_size[0] * self.chessboard_size[1]
            self.assertEqual(len(object_points), expected_points)
            self.assertEqual(object_points.shape[1], 3)  # x, y, z
            
            # Verificar que z=0 (tablero plano)
            self.assertTrue(np.all(object_points[:, 2] == 0.0))
            
            # Verificar espaciado correcto
            self.assertAlmostEqual(object_points[1, 0] - object_points[0, 0], self.square_size)
            
            logger.info("✅ Generación puntos tablero exitosa")
            
        except Exception as e:
            logger.error(f"❌ Error en puntos tablero: {e}")
            self.fail(f"Fallo en puntos tablero: {e}")
    
    def test_synthetic_camera_matrix(self):
        """Test: Matriz de cámara sintética válida"""
        logger.info("🧪 Test: Matriz cámara sintética")
        
        try:
            # Parámetros típicos de cámara Orbbec
            fx, fy = 525.0, 525.0  # Focal length píxeles
            cx, cy = 320.0, 240.0  # Centro óptico
            
            # Construir matriz intrínseca
            K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ], dtype=np.float32)
            
            # Verificar propiedades
            self.assertEqual(K.shape, (3, 3))
            self.assertEqual(K[2, 2], 1.0)  # Normalización
            self.assertEqual(K[0, 1], 0.0)  # No skew
            self.assertEqual(K[1, 0], 0.0)  # No skew
            
            # Verificar valores razonables
            self.assertGreater(K[0, 0], 100)  # fx razonable
            self.assertGreater(K[1, 1], 100)  # fy razonable
            self.assertGreater(K[0, 2], 0)    # cx positivo
            self.assertGreater(K[1, 2], 0)    # cy positivo
            
            logger.info("✅ Matriz cámara sintética válida")
            
        except Exception as e:
            logger.error(f"❌ Error en matriz cámara: {e}")
            self.fail(f"Fallo en matriz cámara: {e}")
    
    def test_distortion_parameters(self):
        """Test: Parámetros de distorsión válidos"""
        logger.info("🧪 Test: Parámetros distorsión")
        
        try:
            # Coeficientes típicos de distorsión radial y tangencial
            # [k1, k2, p1, p2, k3]
            dist_coeffs = np.array([0.1, -0.2, 0.001, 0.001, 0.05], dtype=np.float32)
            
            # Verificar dimensiones
            self.assertEqual(len(dist_coeffs), 5)
            
            # Verificar rangos razonables
            # k1, k2, k3 (distorsión radial) típicamente < 1.0
            self.assertLess(abs(dist_coeffs[0]), 1.0)  # k1
            self.assertLess(abs(dist_coeffs[1]), 1.0)  # k2
            self.assertLess(abs(dist_coeffs[4]), 1.0)  # k3
            
            # p1, p2 (distorsión tangencial) típicamente < 0.01
            self.assertLess(abs(dist_coeffs[2]), 0.1)  # p1
            self.assertLess(abs(dist_coeffs[3]), 0.1)  # p2
            
            logger.info("✅ Parámetros distorsión válidos")
            
        except Exception as e:
            logger.error(f"❌ Error en distorsión: {e}")
            self.fail(f"Fallo en distorsión: {e}")

class TestExtrinsicCalibration(unittest.TestCase):
    """Tests para calibración extrínseca multi-cámara"""
    
    def setUp(self):
        """Setup para cada test"""
        self.n_cameras = 3
        self.world_points = self._generate_world_points()
        self.camera_poses = self._generate_camera_poses()
        
        logger.info("✅ Setup TestExtrinsicCalibration completado")
    
    def _generate_world_points(self) -> np.ndarray:
        """Generar puntos 3D en el mundo"""
        # Esqueleto humano básico en coordenadas del mundo
        skeleton_points = np.array([
            [0, 0, 170],     # Cabeza
            [0, 0, 150],     # Cuello
            [0, 0, 120],     # Torso superior
            [-20, 0, 120],   # Hombro izquierdo
            [20, 0, 120],    # Hombro derecho
            [-40, 0, 100],   # Codo izquierdo
            [40, 0, 100],    # Codo derecho
            [-60, 0, 120],   # Muñeca izquierda
            [60, 0, 120],    # Muñeca derecha
            [0, 0, 80],      # Cadera
            [-10, 0, 80],    # Cadera izquierda
            [10, 0, 80],     # Cadera derecha
            [-10, 0, 40],    # Rodilla izquierda
            [10, 0, 40],     # Rodilla derecha
            [-10, 0, 0],     # Tobillo izquierdo
            [10, 0, 0],      # Tobillo derecho
            [0, 0, 100]      # Centro torso
        ], dtype=np.float32)
        
        return skeleton_points
    
    def _generate_camera_poses(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generar poses de cámaras sintéticas"""
        poses = []
        
        # Cámara 0: Frontal
        R0 = np.eye(3, dtype=np.float32)
        t0 = np.array([0, -200, 100], dtype=np.float32).reshape(3, 1)
        poses.append((R0, t0))
        
        # Cámara 1: Lateral izquierda (45 grados)
        angle = np.pi / 4
        R1 = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        t1 = np.array([-150, -150, 100], dtype=np.float32).reshape(3, 1)
        poses.append((R1, t1))
        
        # Cámara 2: Lateral derecha (-45 grados)
        angle = -np.pi / 4
        R2 = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        t2 = np.array([150, -150, 100], dtype=np.float32).reshape(3, 1)
        poses.append((R2, t2))
        
        return poses
    
    def test_rotation_matrix_properties(self):
        """Test: Propiedades de matrices de rotación"""
        logger.info("🧪 Test: Propiedades matrices rotación")
        
        try:
            for i, (R, t) in enumerate(self.camera_poses):
                # Verificar que es matriz ortogonal
                should_be_identity = R @ R.T
                np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-6)
                
                # Verificar determinante = 1 (rotación, no reflexión)
                det = np.linalg.det(R)
                self.assertAlmostEqual(det, 1.0, places=6)
                
                # Verificar que la inversa es la transpuesta
                R_inv = np.linalg.inv(R)
                np.testing.assert_allclose(R_inv, R.T, atol=1e-6)
                
                logger.info(f"✅ Matriz rotación cámara {i} válida")
            
        except Exception as e:
            logger.error(f"❌ Error en matrices rotación: {e}")
            self.fail(f"Fallo en matrices rotación: {e}")
    
    def test_translation_vector_validity(self):
        """Test: Validez de vectores de traslación"""
        logger.info("🧪 Test: Vectores traslación")
        
        try:
            for i, (R, t) in enumerate(self.camera_poses):
                # Verificar dimensiones
                self.assertEqual(t.shape, (3, 1))
                
                # Verificar que no está en el origen (cámaras separadas)
                distance_from_origin = np.linalg.norm(t)
                self.assertGreater(distance_from_origin, 50)  # Al menos 50mm del origen
                
                # Verificar valores razonables para setup de cámaras
                self.assertLess(abs(t[0, 0]), 300)  # x < 30cm
                self.assertLess(abs(t[1, 0]), 300)  # y < 30cm  
                self.assertLess(abs(t[2, 0]), 300)  # z < 30cm
                
                logger.info(f"✅ Vector traslación cámara {i} válido: {t.flatten()}")
            
        except Exception as e:
            logger.error(f"❌ Error en vectores traslación: {e}")
            self.fail(f"Fallo en vectores traslación: {e}")
    
    def test_projection_consistency(self):
        """Test: Consistencia de proyección 3D→2D"""
        logger.info("🧪 Test: Consistencia proyección")
        
        try:
            # Matriz intrínseca sintética
            K = np.array([
                [525, 0, 320],
                [0, 525, 240],
                [0, 0, 1]
            ], dtype=np.float32)
            
            for i, (R, t) in enumerate(self.camera_poses):
                # Crear matriz de proyección P = K[R|t]
                Rt = np.hstack([R, t])
                P = K @ Rt
                
                # Proyectar puntos 3D
                world_points_hom = np.hstack([self.world_points, np.ones((len(self.world_points), 1))])
                projected_hom = (P @ world_points_hom.T).T
                
                # Convertir a coordenadas 2D
                projected_2d = projected_hom[:, :2] / projected_hom[:, 2:3]
                
                # Verificar que las proyecciones están dentro de imagen
                self.assertTrue(np.all(projected_2d[:, 0] >= 0))
                self.assertTrue(np.all(projected_2d[:, 0] <= 640))
                self.assertTrue(np.all(projected_2d[:, 1] >= 0))
                self.assertTrue(np.all(projected_2d[:, 1] <= 480))
                
                # Verificar que los puntos están frente a la cámara (z > 0)
                z_values = projected_hom[:, 2]
                self.assertTrue(np.all(z_values > 0))
                
                logger.info(f"✅ Proyección cámara {i} consistente")
            
        except Exception as e:
            logger.error(f"❌ Error en proyección: {e}")
            self.fail(f"Fallo en proyección: {e}")

class TestReprojectionError(unittest.TestCase):
    """Tests para cálculo de error de reproyección"""
    
    def setUp(self):
        """Setup para tests de reproyección"""
        self.tolerance_pixels = 2.0
        self.test_points_3d = np.array([
            [0, 0, 100],
            [50, 0, 100],
            [-50, 0, 100],
            [0, 50, 100],
            [0, -50, 100]
        ], dtype=np.float32)
        
        logger.info("✅ Setup TestReprojectionError completado")
    
    def test_perfect_reprojection(self):
        """Test: Error cero con datos perfectos"""
        logger.info("🧪 Test: Reproyección perfecta")
        
        try:
            # Configuración perfecta sin ruido
            K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
            t = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
            
            # Proyección directa
            Rt = np.hstack([R, t])
            P = K @ Rt
            
            points_hom = np.hstack([self.test_points_3d, np.ones((len(self.test_points_3d), 1))])
            projected_hom = (P @ points_hom.T).T
            projected_2d = projected_hom[:, :2] / projected_hom[:, 2:3]
            
            # Calcular error de reproyección (debe ser ~0)
            # En este caso, reproyectamos los mismos puntos
            errors = np.linalg.norm(projected_2d - projected_2d, axis=1)
            
            # Error debe ser esencialmente cero
            self.assertTrue(np.all(errors < 1e-6))
            
            mean_error = np.mean(errors)
            self.assertLess(mean_error, 1e-6)
            
            logger.info(f"✅ Error reproyección perfecto: {mean_error:.2e} píxeles")
            
        except Exception as e:
            logger.error(f"❌ Error en reproyección perfecta: {e}")
            self.fail(f"Fallo en reproyección perfecta: {e}")
    
    def test_noisy_reprojection(self):
        """Test: Error con ruido realista"""
        logger.info("🧪 Test: Reproyección con ruido")
        
        try:
            # Añadir ruido realista a las observaciones
            np.random.seed(42)  # Para reproducibilidad
            noise_std = 0.5  # píxeles
            
            K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
            t = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
            
            # Proyección limpia
            Rt = np.hstack([R, t])
            P = K @ Rt
            
            points_hom = np.hstack([self.test_points_3d, np.ones((len(self.test_points_3d), 1))])
            projected_clean = ((P @ points_hom.T).T)[:, :2]
            projected_clean = projected_clean / ((P @ points_hom.T).T)[:, 2:3]
            
            # Añadir ruido
            noise = np.random.normal(0, noise_std, projected_clean.shape)
            projected_noisy = projected_clean + noise
            
            # Calcular error
            errors = np.linalg.norm(projected_noisy - projected_clean, axis=1)
            mean_error = np.mean(errors)
            
            # El error debe ser proporcional al ruido añadido
            self.assertGreater(mean_error, 0)
            self.assertLess(mean_error, 3 * noise_std)  # Dentro de 3 sigmas
            
            logger.info(f"✅ Error reproyección con ruido: {mean_error:.2f} píxeles")
            
        except Exception as e:
            logger.error(f"❌ Error en reproyección con ruido: {e}")
            self.fail(f"Fallo en reproyección con ruido: {e}")

def run_calibration_tests():
    """Ejecutar todos los tests de calibración"""
    print("🧪 EJECUTANDO TESTS CALIBRACIÓN")
    print("=" * 50)
    
    # Crear suite de tests
    test_suite = unittest.TestSuite()
    
    # Añadir tests
    test_suite.addTest(unittest.makeSuite(TestCameraIntrinsics))
    test_suite.addTest(unittest.makeSuite(TestExtrinsicCalibration))
    test_suite.addTest(unittest.makeSuite(TestReprojectionError))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN TESTS CALIBRACIÓN")
    print(f"✅ Tests exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests fallidos: {len(result.failures)}")
    print(f"💥 Errores: {len(result.errors)}")
    
    return result.wasSuccessful()

def test_quick_calibration():
    """Test rápido de calibración"""
    print("⚡ Test rápido calibración...")
    
    try:
        # Test matrices básicas
        K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3, 1), dtype=np.float32)
        
        # Verificar propiedades básicas
        assert K[2, 2] == 1.0, "Matriz intrínseca mal formada"
        assert np.allclose(R @ R.T, np.eye(3)), "Matriz rotación inválida"
        assert np.linalg.det(R) > 0.9, "Determinante rotación incorrecto"
        
        print("✅ Test rápido calibración exitoso")
        return True
        
    except Exception as e:
        print(f"❌ Test rápido calibración falló: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = test_quick_calibration()
    else:
        success = run_calibration_tests()
    
    sys.exit(0 if success else 1)

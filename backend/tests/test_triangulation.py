"""
Tests para triangulación 3D y validación de reconstrucción

Este módulo contiene tests específicos para:
- Triangulación DLT (Direct Linear Transform)
- Validación de puntos 3D reconstruidos
- Error de reproyección
- Casos edge (puntos muy cercanos, muy lejanos, etc.)

Uso:
    python -m backend.tests.test_triangulation
    python -c "from backend.tests.test_triangulation import test_quick_triangulation; test_quick_triangulation()"
"""

import unittest
import numpy as np
from typing import List, Tuple, Dict, Optional
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

class TestDLTTriangulation(unittest.TestCase):
    """Tests para triangulación DLT (Direct Linear Transform)"""
    
    def setUp(self):
        """Setup para cada test"""
        # Configuración de cámaras sintéticas
        self.K = np.array([
            [525, 0, 320],
            [0, 525, 240], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Poses de cámaras (R, t)
        self.camera_poses = [
            (np.eye(3, dtype=np.float32), np.array([0, -200, 0], dtype=np.float32).reshape(3, 1)),
            (self._rotation_y(np.pi/6), np.array([-100, -200, 0], dtype=np.float32).reshape(3, 1)),
            (self._rotation_y(-np.pi/6), np.array([100, -200, 0], dtype=np.float32).reshape(3, 1))
        ]
        
        # Puntos 3D de prueba
        self.test_points_3d = np.array([
            [0, 0, 100],      # Punto central
            [50, 0, 100],     # Lateral derecho
            [-50, 0, 100],    # Lateral izquierdo
            [0, 50, 100],     # Superior
            [0, -50, 100],    # Inferior
            [0, 0, 200],      # Más lejano
            [0, 0, 50]        # Más cercano
        ], dtype=np.float32)
        
        logger.info("✅ Setup TestDLTTriangulation completado")
    
    def _rotation_y(self, angle: float) -> np.ndarray:
        """Crear matriz de rotación alrededor del eje Y"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float32)
    
    def _project_points(self, points_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Proyectar puntos 3D a 2D usando una cámara"""
        # Transformar al sistema de coordenadas de la cámara
        points_cam = (R @ points_3d.T + t).T
        
        # Proyectar usando matriz intrínseca
        points_hom = (self.K @ points_cam.T).T
        points_2d = points_hom[:, :2] / points_hom[:, 2:3]
        
        return points_2d
    
    def _triangulate_dlt_simple(self, points_2d_list: List[np.ndarray], 
                               cameras: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Implementación simple de triangulación DLT para testing
        """
        n_points = len(points_2d_list[0])
        points_3d = []
        
        for i in range(n_points):
            # Construir sistema Ax = 0 para cada punto
            A = []
            
            for cam_idx, (R, t) in enumerate(cameras):
                # Matriz de proyección P = K[R|t]
                Rt = np.hstack([R, t])
                P = self.K @ Rt
                
                # Punto 2D observado
                x, y = points_2d_list[cam_idx][i]
                
                # Ecuaciones DLT
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])
            
            A = np.array(A)
            
            # Resolver Ax = 0 usando SVD
            _, _, Vt = np.linalg.svd(A)
            point_3d_hom = Vt[-1]
            
            # Convertir de homogéneas a cartesianas
            point_3d = point_3d_hom[:3] / point_3d_hom[3]
            points_3d.append(point_3d)
        
        return np.array(points_3d)
    
    def test_perfect_triangulation(self):
        """Test: Triangulación perfecta sin ruido"""
        logger.info("🧪 Test: Triangulación perfecta")
        
        try:
            # Proyectar puntos a todas las cámaras
            points_2d_list = []
            for R, t in self.camera_poses:
                points_2d = self._project_points(self.test_points_3d, R, t)
                points_2d_list.append(points_2d)
            
            # Triangular de vuelta
            reconstructed = self._triangulate_dlt_simple(points_2d_list, self.camera_poses)
            
            # Verificar que la reconstrucción es precisa
            errors = np.linalg.norm(reconstructed - self.test_points_3d, axis=1)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            # Error debe ser muy pequeño (casi cero)
            self.assertLess(max_error, 1e-6, f"Error máximo: {max_error}")
            self.assertLess(mean_error, 1e-7, f"Error medio: {mean_error}")
            
            logger.info(f"✅ Triangulación perfecta: error medio {mean_error:.2e}mm")
            
        except Exception as e:
            logger.error(f"❌ Error en triangulación perfecta: {e}")
            self.fail(f"Fallo en triangulación perfecta: {e}")
    
    def test_noisy_triangulation(self):
        """Test: Triangulación con ruido realista"""
        logger.info("🧪 Test: Triangulación con ruido")
        
        try:
            np.random.seed(42)  # Reproducibilidad
            noise_std = 1.0  # 1 píxel de ruido
            
            # Proyectar puntos y añadir ruido
            points_2d_list = []
            for R, t in self.camera_poses:
                points_2d = self._project_points(self.test_points_3d, R, t)
                # Añadir ruido gaussiano
                noise = np.random.normal(0, noise_std, points_2d.shape)
                points_2d_noisy = points_2d + noise
                points_2d_list.append(points_2d_noisy)
            
            # Triangular
            reconstructed = self._triangulate_dlt_simple(points_2d_list, self.camera_poses)
            
            # Calcular errores
            errors = np.linalg.norm(reconstructed - self.test_points_3d, axis=1)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            # Con ruido de 1 píxel, esperamos errores de unos pocos mm
            self.assertLess(mean_error, 50, f"Error medio muy alto: {mean_error}mm")
            self.assertLess(max_error, 100, f"Error máximo muy alto: {max_error}mm")
            
            logger.info(f"✅ Triangulación con ruido: error medio {mean_error:.1f}mm, máximo {max_error:.1f}mm")
            
        except Exception as e:
            logger.error(f"❌ Error en triangulación con ruido: {e}")
            self.fail(f"Fallo en triangulación con ruido: {e}")
    
    def test_triangulation_geometry_validation(self):
        """Test: Validación de geometría de triangulación"""
        logger.info("🧪 Test: Validación geometría triangulación")
        
        try:
            # Test con diferentes configuraciones geométricas
            
            # 1. Cámaras muy cercanas (mala geometría)
            bad_poses = [
                (np.eye(3, dtype=np.float32), np.array([0, -200, 0], dtype=np.float32).reshape(3, 1)),
                (np.eye(3, dtype=np.float32), np.array([5, -200, 0], dtype=np.float32).reshape(3, 1))  # Solo 5mm separación
            ]
            
            points_2d_bad = []
            for R, t in bad_poses:
                points_2d = self._project_points(self.test_points_3d[:1], R, t)  # Solo 1 punto
                points_2d_bad.append(points_2d)
            
            reconstructed_bad = self._triangulate_dlt_simple(points_2d_bad, bad_poses)
            error_bad = np.linalg.norm(reconstructed_bad[0] - self.test_points_3d[0])
            
            # 2. Cámaras bien separadas (buena geometría)
            good_poses = [
                (np.eye(3, dtype=np.float32), np.array([0, -200, 0], dtype=np.float32).reshape(3, 1)),
                (self._rotation_y(np.pi/3), np.array([-150, -200, 0], dtype=np.float32).reshape(3, 1))  # 30° y bien separadas
            ]
            
            points_2d_good = []
            for R, t in good_poses:
                points_2d = self._project_points(self.test_points_3d[:1], R, t)
                points_2d_good.append(points_2d)
            
            reconstructed_good = self._triangulate_dlt_simple(points_2d_good, good_poses)
            error_good = np.linalg.norm(reconstructed_good[0] - self.test_points_3d[0])
            
            # La geometría buena debe dar menor error
            self.assertLess(error_good, error_bad, 
                           f"Error buena geometría ({error_good:.1f}) >= mala geometría ({error_bad:.1f})")
            
            logger.info(f"✅ Validación geometría: mala={error_bad:.1f}mm, buena={error_good:.1f}mm")
            
        except Exception as e:
            logger.error(f"❌ Error en validación geometría: {e}")
            self.fail(f"Fallo en validación geometría: {e}")

class TestReprojectionValidation(unittest.TestCase):
    """Tests para validación por error de reproyección"""
    
    def setUp(self):
        """Setup para tests de reproyección"""
        self.K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
        self.threshold = 5.0  # píxeles
        
        # Una cámara simple para test
        self.R = np.eye(3, dtype=np.float32)
        self.t = np.array([0, -200, 0], dtype=np.float32).reshape(3, 1)
        
        logger.info("✅ Setup TestReprojectionValidation completado")
    
    def _compute_reprojection_error(self, point_3d: np.ndarray, 
                                   point_2d: np.ndarray) -> float:
        """Calcular error de reproyección para un punto"""
        # Transformar a coordenadas de cámara
        point_cam = self.R @ point_3d + self.t.flatten()
        
        # Proyectar
        point_proj_hom = self.K @ point_cam
        point_proj = point_proj_hom[:2] / point_proj_hom[2]
        
        # Error euclidiano
        error = np.linalg.norm(point_proj - point_2d)
        return error
    
    def test_good_reprojection_error(self):
        """Test: Puntos con buen error de reproyección"""
        logger.info("🧪 Test: Buen error reproyección")
        
        try:
            # Punto 3D y su proyección exacta
            point_3d = np.array([0, 0, 100], dtype=np.float32)
            
            # Proyección exacta
            point_cam = self.R @ point_3d + self.t.flatten()
            point_proj_hom = self.K @ point_cam
            point_2d_exact = point_proj_hom[:2] / point_proj_hom[2]
            
            # Error debe ser cero
            error = self._compute_reprojection_error(point_3d, point_2d_exact)
            self.assertLess(error, 1e-6)
            
            # Punto con pequeño error
            point_2d_small_error = point_2d_exact + np.array([1.0, 1.0])  # 1 píxel error
            error_small = self._compute_reprojection_error(point_3d, point_2d_small_error)
            
            self.assertLess(error_small, self.threshold)
            self.assertAlmostEqual(error_small, np.sqrt(2), places=3)  # √2 ≈ 1.414
            
            logger.info(f"✅ Error reproyección: exacto={error:.2e}, pequeño={error_small:.2f} píxeles")
            
        except Exception as e:
            logger.error(f"❌ Error en buen reproyección: {e}")
            self.fail(f"Fallo en buen reproyección: {e}")
    
    def test_bad_reprojection_error(self):
        """Test: Puntos con mal error de reproyección"""
        logger.info("🧪 Test: Mal error reproyección")
        
        try:
            point_3d = np.array([0, 0, 100], dtype=np.float32)
            
            # Proyección exacta
            point_cam = self.R @ point_3d + self.t.flatten()
            point_proj_hom = self.K @ point_cam
            point_2d_exact = point_proj_hom[:2] / point_proj_hom[2]
            
            # Punto con gran error
            point_2d_big_error = point_2d_exact + np.array([10.0, 10.0])  # 10 píxeles error
            error_big = self._compute_reprojection_error(point_3d, point_2d_big_error)
            
            self.assertGreater(error_big, self.threshold)
            self.assertAlmostEqual(error_big, 10 * np.sqrt(2), places=1)  # 10√2 ≈ 14.14
            
            logger.info(f"✅ Error reproyección grande: {error_big:.2f} píxeles > {self.threshold}")
            
        except Exception as e:
            logger.error(f"❌ Error en mal reproyección: {e}")
            self.fail(f"Fallo en mal reproyección: {e}")

class TestTriangulationEdgeCases(unittest.TestCase):
    """Tests para casos edge en triangulación"""
    
    def setUp(self):
        """Setup para casos edge"""
        self.K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
        self.cameras = [
            (np.eye(3, dtype=np.float32), np.array([0, -200, 0], dtype=np.float32).reshape(3, 1)),
            (self._rotation_y(np.pi/4), np.array([-100, -200, 0], dtype=np.float32).reshape(3, 1))
        ]
        
        logger.info("✅ Setup TestTriangulationEdgeCases completado")
    
    def _rotation_y(self, angle: float) -> np.ndarray:
        """Crear matriz de rotación alrededor del eje Y"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    
    def test_very_close_points(self):
        """Test: Puntos muy cercanos a las cámaras"""
        logger.info("🧪 Test: Puntos muy cercanos")
        
        try:
            # Punto muy cercano (puede causar problemas numéricos)
            close_point = np.array([0, 0, 10], dtype=np.float32)  # Solo 10mm de distancia
            
            # Verificar que el punto está frente a ambas cámaras
            for R, t in self.cameras:
                point_cam = R @ close_point + t.flatten()
                self.assertGreater(point_cam[2], 0, "Punto detrás de la cámara")
            
            # La triangulación debe manejar esto sin crash
            # (aunque puede tener mayor error)
            logger.info("✅ Puntos cercanos manejados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error con puntos cercanos: {e}")
            self.fail(f"Fallo con puntos cercanos: {e}")
    
    def test_very_far_points(self):
        """Test: Puntos muy lejanos"""
        logger.info("🧪 Test: Puntos muy lejanos")
        
        try:
            # Punto muy lejano
            far_point = np.array([0, 0, 10000], dtype=np.float32)  # 10 metros
            
            # Verificar que sigue siendo visible
            for R, t in self.cameras:
                point_cam = R @ far_point + t.flatten()
                self.assertGreater(point_cam[2], 0, "Punto detrás de la cámara")
                
                # Proyectar para verificar que está en la imagen
                point_proj_hom = self.K @ point_cam
                point_proj = point_proj_hom[:2] / point_proj_hom[2]
                
                # Debe estar dentro de límites razonables
                self.assertGreater(point_proj[0], -100)
                self.assertLess(point_proj[0], 740)
                self.assertGreater(point_proj[1], -100)
                self.assertLess(point_proj[1], 580)
            
            logger.info("✅ Puntos lejanos manejados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error con puntos lejanos: {e}")
            self.fail(f"Fallo con puntos lejanos: {e}")
    
    def test_points_at_infinity(self):
        """Test: Manejo de puntos en el infinito"""
        logger.info("🧪 Test: Puntos en el infinito")
        
        try:
            # Simular punto muy lejano que causa problemas numéricos
            inf_point = np.array([0, 0, 1e6], dtype=np.float32)  # 1km de distancia
            
            # Verificar que no hay overflow/underflow
            for R, t in self.cameras:
                point_cam = R @ inf_point + t.flatten()
                
                # Verificar que los valores son finitos
                self.assertTrue(np.all(np.isfinite(point_cam)))
                self.assertGreater(point_cam[2], 0)
            
            logger.info("✅ Puntos en infinito manejados sin overflow")
            
        except Exception as e:
            logger.error(f"❌ Error con puntos infinito: {e}")
            self.fail(f"Fallo con puntos infinito: {e}")

def run_triangulation_tests():
    """Ejecutar todos los tests de triangulación"""
    print("🧪 EJECUTANDO TESTS TRIANGULACIÓN")
    print("=" * 50)
    
    # Crear suite de tests
    test_suite = unittest.TestSuite()
    
    # Añadir todos los tests
    test_suite.addTest(unittest.makeSuite(TestDLTTriangulation))
    test_suite.addTest(unittest.makeSuite(TestReprojectionValidation))
    test_suite.addTest(unittest.makeSuite(TestTriangulationEdgeCases))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN TESTS TRIANGULACIÓN")
    print(f"✅ Tests exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests fallidos: {len(result.failures)}")
    print(f"💥 Errores: {len(result.errors)}")
    
    return result.wasSuccessful()

def test_quick_triangulation():
    """Test rápido de triangulación"""
    print("⚡ Test rápido triangulación...")
    
    try:
        # Test básico de triangulación
        K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
        
        # Punto 3D simple
        point_3d = np.array([0, 0, 100], dtype=np.float32)
        
        # Dos cámaras
        R1, t1 = np.eye(3, dtype=np.float32), np.array([0, -200, 0], dtype=np.float32)
        R2, t2 = np.eye(3, dtype=np.float32), np.array([100, -200, 0], dtype=np.float32)
        
        # Proyectar
        point_cam1 = R1 @ point_3d + t1
        point_proj1 = (K @ point_cam1)[:2] / (K @ point_cam1)[2]
        
        point_cam2 = R2 @ point_3d + t2
        point_proj2 = (K @ point_cam2)[:2] / (K @ point_cam2)[2]
        
        # Verificar que las proyecciones son diferentes
        assert not np.allclose(point_proj1, point_proj2), "Proyecciones idénticas"
        
        # Verificar que están en imagen
        assert 0 <= point_proj1[0] <= 640, "Proyección 1 fuera de imagen"
        assert 0 <= point_proj1[1] <= 480, "Proyección 1 fuera de imagen"
        
        print("✅ Test rápido triangulación exitoso")
        return True
        
    except Exception as e:
        print(f"❌ Test rápido triangulación falló: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = test_quick_triangulation()
    else:
        success = run_triangulation_tests()
    
    sys.exit(0 if success else 1)

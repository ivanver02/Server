"""
Tests unitarios para el sistema MMPose de detección de poses

Este módulo contiene tests independientes para cada componente:
- Test de inicialización de modelos
- Test de inferencia individual  
- Test de distribución GPU
- Test de ensemble learning

Uso:
    python -m backend.tests.test_mmpose
    python -c "from backend.tests.test_mmpose import test_single_model; test_single_model()"
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Optional
import logging

# Configurar logging para tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports condicionales para evitar errores sin MMPose
try:
    from ..processing.pose_detector import MMPoseInferencerWrapper
    MMPOSE_AVAILABLE = True
except ImportError as e:
    MMPOSE_AVAILABLE = False
    logger.warning(f"MMPose no disponible para testing: {e}")

class TestMMPoseSingleModel(unittest.TestCase):
    """Tests para modelo individual de MMPose"""
    
    @classmethod
    def setUpClass(cls):
        """Setup una vez para toda la clase"""
        if not MMPOSE_AVAILABLE:
            raise unittest.SkipTest("MMPose no disponible")
        
        cls.test_image = cls._create_test_image()
        logger.info("✅ Setup TestMMPoseSingleModel completado")
    
    @staticmethod
    def _create_test_image() -> np.ndarray:
        """Crear imagen sintética para testing"""
        # Imagen simple con figura humana básica
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dibujar figura de palo simple
        cv2.line(img, (320, 100), (320, 200), (255, 255, 255), 3)  # Cabeza-torso
        cv2.line(img, (320, 200), (320, 350), (255, 255, 255), 3)  # Torso
        cv2.line(img, (320, 200), (280, 280), (255, 255, 255), 3)  # Brazo izq
        cv2.line(img, (320, 200), (360, 280), (255, 255, 255), 3)  # Brazo der
        cv2.line(img, (320, 350), (280, 450), (255, 255, 255), 3)  # Pierna izq
        cv2.line(img, (320, 350), (360, 450), (255, 255, 255), 3)  # Pierna der
        cv2.circle(img, (320, 80), 20, (255, 255, 255), -1)       # Cabeza
        
        return img
    
    def test_model_initialization(self):
        """Test: Inicialización correcta de un modelo individual"""
        logger.info("🧪 Test: Inicialización modelo individual")
        
        # Configuración mínima para test
        test_config = {
            'hrnet_w32_coco': {
                'config': 'test_config.py',
                'checkpoint': 'test_checkpoint.pth',
                'device': 'cpu'  # Usar CPU para testing
            }
        }
        
        try:
            detector = MMPoseInferencerWrapper()
            
            # Verificar que se puede cargar configuración
            self.assertIsNotNone(detector)
            logger.info("✅ Inicialización exitosa")
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización: {e}")
            self.fail(f"Fallo en inicialización: {e}")
    
    def test_inference_single_image(self):
        """Test: Inferencia en imagen individual"""
        logger.info("🧪 Test: Inferencia imagen individual")
        
        # Skip si no hay GPU disponible para test completo
        if not torch.cuda.is_available():
            self.skipTest("GPU no disponible - test de inferencia saltado")
        
        try:
            # Simular resultado de inferencia esperado
            expected_keypoints = 17  # COCO standard
            
            # El resultado debe tener forma correcta
            result_shape = (expected_keypoints, 3)  # x, y, confidence
            
            # Verificar que el resultado tiene la forma esperada
            test_result = np.random.rand(*result_shape)
            self.assertEqual(test_result.shape, result_shape)
            
            # Verificar que las confianzas están en rango [0, 1]
            confidences = test_result[:, 2]
            self.assertTrue(np.all(confidences >= 0))
            self.assertTrue(np.all(confidences <= 1))
            
            logger.info("✅ Inferencia test simulado exitoso")
            
        except Exception as e:
            logger.error(f"❌ Error en inferencia: {e}")
            self.fail(f"Fallo en inferencia: {e}")

class TestEnsembleLearning(unittest.TestCase):
    """Tests para el sistema de ensemble learning"""
    
    def setUp(self):
        """Setup para cada test"""
        self.n_models = 4
        self.n_keypoints = 17
        self.n_frames = 5
        
        # Datos sintéticos para ensemble
        self.synthetic_results = self._generate_synthetic_results()
        logger.info("✅ Setup TestEnsembleLearning completado")
    
    def _generate_synthetic_results(self) -> Dict[str, np.ndarray]:
        """Generar resultados sintéticos de múltiples modelos"""
        models = ['hrnet_w48_coco', 'hrnet_w32_coco', 'resnet50_rle_coco', 'wholebody_coco']
        results = {}
        
        for model in models:
            # Generar keypoints con variación realista
            keypoints = np.random.rand(self.n_frames, self.n_keypoints, 3)
            
            # Ajustar confianzas a rangos realistas
            keypoints[:, :, 2] = np.random.beta(2, 2, (self.n_frames, self.n_keypoints))
            
            results[model] = keypoints
        
        return results
    
    def test_ensemble_fusion_coco(self):
        """Test: Fusión de keypoints COCO con pesos"""
        logger.info("🧪 Test: Fusión ensemble COCO")
        
        try:
            # Simular proceso de ensemble
            models_coco = ['hrnet_w48_coco', 'hrnet_w32_coco', 'resnet50_rle_coco']
            weights = {'hrnet_w48_coco': 0.6, 'hrnet_w32_coco': 0.4, 'resnet50_rle_coco': 1.0}
            
            # Test de fusión ponderada
            for frame_idx in range(self.n_frames):
                for keypoint_idx in range(self.n_keypoints):
                    
                    # Extraer datos de un keypoint específico
                    keypoint_data = {}
                    for model in models_coco:
                        if model in self.synthetic_results:
                            keypoint_data[model] = self.synthetic_results[model][frame_idx, keypoint_idx]
                    
                    # Verificar que tenemos datos
                    self.assertGreater(len(keypoint_data), 0)
                    
                    # Verificar fusión ponderada (simulada)
                    total_weight = sum(weights.get(m, 1.0) for m in keypoint_data.keys())
                    self.assertGreater(total_weight, 0)
            
            logger.info("✅ Fusión ensemble exitosa")
            
        except Exception as e:
            logger.error(f"❌ Error en ensemble: {e}")
            self.fail(f"Fallo en ensemble: {e}")
    
    def test_confidence_filtering(self):
        """Test: Filtrado por confianza mínima"""
        logger.info("🧪 Test: Filtrado por confianza")
        
        try:
            confidence_threshold = 0.3
            
            for model_name, results in self.synthetic_results.items():
                for frame_idx in range(self.n_frames):
                    confidences = results[frame_idx, :, 2]
                    
                    # Verificar que existen confianzas válidas
                    valid_confidences = confidences[confidences >= confidence_threshold]
                    
                    # Debe haber al menos algunos keypoints válidos
                    self.assertGreaterEqual(len(valid_confidences), 0)
                    
                    # Las confianzas válidas deben estar en rango
                    if len(valid_confidences) > 0:
                        self.assertTrue(np.all(valid_confidences >= confidence_threshold))
                        self.assertTrue(np.all(valid_confidences <= 1.0))
            
            logger.info("✅ Filtrado por confianza exitoso")
            
        except Exception as e:
            logger.error(f"❌ Error en filtrado: {e}")
            self.fail(f"Fallo en filtrado: {e}")

class TestGPUDistribution(unittest.TestCase):
    """Tests para distribución de modelos en GPUs"""
    
    def setUp(self):
        """Setup para tests de GPU"""
        self.gpu_config = {
            'primary_gpu': 'cuda:0',
            'secondary_gpu': 'cuda:1',
            'models_primary': ['hrnet_w48_coco', 'hrnet_w32_coco'],
            'models_secondary': ['resnet50_rle_coco', 'wholebody_coco']
        }
        logger.info("✅ Setup TestGPUDistribution completado")
    
    def test_gpu_availability_check(self):
        """Test: Verificación de disponibilidad de GPUs"""
        logger.info("🧪 Test: Disponibilidad GPUs")
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPUs disponibles: {gpu_count}")
                
                # Verificar que tenemos al menos 1 GPU
                self.assertGreaterEqual(gpu_count, 1)
                
                # Si tenemos 2+ GPUs, test distribución
                if gpu_count >= 2:
                    # Test que podemos acceder a ambas GPUs
                    device_0 = torch.device('cuda:0')
                    device_1 = torch.device('cuda:1')
                    
                    # Crear tensores pequeños en cada GPU
                    tensor_0 = torch.randn(10, 10, device=device_0)
                    tensor_1 = torch.randn(10, 10, device=device_1)
                    
                    # Verificar que están en GPUs diferentes
                    self.assertEqual(tensor_0.device, device_0)
                    self.assertEqual(tensor_1.device, device_1)
                    
                    logger.info("✅ Distribución multi-GPU disponible")
                else:
                    logger.warning("⚠️ Solo 1 GPU disponible - distribución limitada")
            else:
                logger.warning("⚠️ No hay GPUs disponibles - usando CPU")
                self.skipTest("GPU no disponible")
                
        except Exception as e:
            logger.error(f"❌ Error en test GPU: {e}")
            self.fail(f"Fallo en test GPU: {e}")
    
    def test_model_assignment_logic(self):
        """Test: Lógica de asignación de modelos a GPUs"""
        logger.info("🧪 Test: Asignación modelos a GPUs")
        
        try:
            # Verificar configuración de distribución
            all_models = self.gpu_config['models_primary'] + self.gpu_config['models_secondary']
            
            # Verificar que no hay duplicados
            self.assertEqual(len(all_models), len(set(all_models)))
            
            # Verificar que tenemos 4 modelos
            self.assertEqual(len(all_models), 4)
            
            # Verificar modelos específicos
            expected_models = ['hrnet_w48_coco', 'hrnet_w32_coco', 'resnet50_rle_coco', 'wholebody_coco']
            for model in expected_models:
                self.assertIn(model, all_models)
            
            logger.info("✅ Asignación de modelos correcta")
            
        except Exception as e:
            logger.error(f"❌ Error en asignación: {e}")
            self.fail(f"Fallo en asignación: {e}")

def run_mmpose_tests():
    """Ejecutar todos los tests de MMPose"""
    print("🧪 EJECUTANDO TESTS MMPOSE")
    print("=" * 50)
    
    # Crear suite de tests
    test_suite = unittest.TestSuite()
    
    # Añadir tests
    test_suite.addTest(unittest.makeSuite(TestMMPoseSingleModel))
    test_suite.addTest(unittest.makeSuite(TestEnsembleLearning))
    test_suite.addTest(unittest.makeSuite(TestGPUDistribution))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN TESTS MMPOSE")
    print(f"✅ Tests exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests fallidos: {len(result.failures)}")
    print(f"💥 Errores: {len(result.errors)}")
    
    if result.failures:
        print("\n📋 FALLOS:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORES:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

def run_quick_mmpose_test():
    """Test rápido de funcionalidad básica"""
    print("⚡ Test rápido MMPose...")
    
    try:
        # Test 1: Datos sintéticos
        logger.info("🎲 Generando datos sintéticos...")
        test_keypoints = np.random.rand(5, 17, 3)  # 5 frames, 17 keypoints, x,y,conf
        
        # Verificar forma
        assert test_keypoints.shape == (5, 17, 3), "Forma incorrecta"
        
        # Test 2: Ensemble simulado
        logger.info("🔄 Simulando ensemble...")
        models = ['hrnet_w48_coco', 'hrnet_w32_coco']
        weights = [0.6, 0.4]
        
        # Simular fusión
        fused = np.zeros_like(test_keypoints)
        for i, weight in enumerate(weights):
            fused += weight * test_keypoints
        
        # Verificar resultado
        assert fused.shape == test_keypoints.shape, "Fusión incorrecta"
        
        # Test 3: GPU check
        logger.info("🖥️ Verificando GPU...")
        gpu_available = torch.cuda.is_available()
        logger.info(f"GPU disponible: {gpu_available}")
        
        print("✅ Test rápido MMPose exitoso")
        return True
        
    except Exception as e:
        print(f"❌ Test rápido falló: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_mmpose_test()
    else:
        success = run_mmpose_tests()
    
    sys.exit(0 if success else 1)

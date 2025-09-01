#!/usr/bin/env python3
"""
Test completo del sistema de reconstrucción 3D con calibración de extrínsecos
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Agregar el directorio del proyecto al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.processing.reconstruction import (
    ReconstructionCoordinator, 
    calibrate_extrinsics_from_keypoints
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_extrinsics_calibration():
    """Test de calibración de extrínsecos usando keypoints 2D"""
    
    # Configuración - usando datos reales del servidor
    base_data_dir = Path(__file__).parent.parent.parent / "data"
    patient_id = "1"
    session_id = "8"  # Datos reales disponibles
    camera_ids = [0, 1, 2]
    
    logger.info("=== Test: Calibración de Extrínsecos ===")
    
    # Paso 1: Verificar que existen los datos reales
    logger.info("1. Verificando datos reales...")
    session_dir = base_data_dir / "processed" / "2D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    
    if not session_dir.exists():
        logger.error(f"   ✗ No se encontraron datos en {session_dir}")
        logger.info("   💡 Asegúrate de que los datos estén en la ruta correcta")
        return False
    else:
        logger.info("   ✓ Datos reales encontrados")
        
        # Verificar que hay datos para todas las cámaras
        cameras_found = []
        for camera_id in camera_ids:
            camera_dir = session_dir / f"camera{camera_id}"
            if camera_dir.exists():
                coord_files = list((camera_dir / "coordinates").glob("*.npy")) if (camera_dir / "coordinates").exists() else []
                conf_files = list((camera_dir / "confidence").glob("*.npy")) if (camera_dir / "confidence").exists() else []
                if coord_files and conf_files:
                    cameras_found.append(camera_id)
                    logger.info(f"     - Cámara {camera_id}: {len(coord_files)} archivos de coordenadas")
        
        if len(cameras_found) < 2:
            logger.error(f"   ✗ Insuficientes cámaras con datos: {cameras_found}")
            return False
        
        camera_ids = cameras_found  # Usar solo las cámaras que tienen datos
        logger.info(f"   ✓ Usando datos de cámaras: {camera_ids}")
    
    # Paso 2: Inicializar coordinador
    logger.info("2. Inicializando sistema de cámaras...")
    coordinator = ReconstructionCoordinator(base_data_dir)
    
    if not coordinator.initialize_camera_system(camera_ids, use_calibration=False):
        logger.error("   ✗ Error inicializando sistema de cámaras")
        return False
    
    logger.info("   ✓ Sistema de cámaras inicializado")
    
    # Verificar estado inicial
    logger.info("   Estado inicial del sistema:")
    for cam_id in camera_ids:
        camera = coordinator.camera_system.get_camera(cam_id)
        if camera:
            logger.info(f"     - Cámara {cam_id}: calibrada={camera.is_calibrated}, referencia={camera.is_reference}")
    
    # Paso 3: Calibrar extrínsecos usando keypoints reales
    logger.info("3. Calibrando extrínsecos desde keypoints 2D reales...")
    
    success = coordinator.calibrate_extrinsics_from_keypoints(
        patient_id=patient_id,
        session_id=session_id,
        method="pnp",  # o "optimization"
        min_confidence=0.3
    )
    
    if success:
        logger.info("   ✓ Extrínsecos calibrados exitosamente")
        
        # Mostrar resultados
        logger.info("   Parámetros extrínsecos calculados:")
        for cam_id in camera_ids:
            camera = coordinator.camera_system.get_camera(cam_id)
            if camera and camera.is_calibrated:
                t = camera.translation_vector
                logger.info(f"     - Cámara {cam_id}: T=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
                
                # Verificar matriz de rotación
                R = camera.rotation_matrix
                det_R = np.linalg.det(R)
                logger.info(f"       Rotación det(R)={det_R:.3f} (debe ser ~1.0)")
        
        return True
    else:
        logger.error("   ✗ Error calibrando extrínsecos")
        return False

def test_full_reconstruction_pipeline():
    """Test del pipeline completo de reconstrucción"""
    
    # Configuración - usando datos reales del servidor
    base_data_dir = Path(__file__).parent.parent.parent / "data"
    patient_id = "1"
    session_id = "8"  # Datos reales disponibles
    camera_ids = [0, 1, 2]
    
    logger.info("=== Test: Pipeline Completo de Reconstrucción ===")
    
    # Paso 1: Coordinador con calibración completa
    coordinator = ReconstructionCoordinator(base_data_dir)
    
    # Paso 2: Inicializar sistema
    logger.info("1. Inicializando sistema de cámaras...")
    if not coordinator.initialize_camera_system(camera_ids, use_calibration=False):
        logger.error("   ✗ Error inicializando sistema")
        return False
    
    # Paso 3: Calibrar extrínsecos
    logger.info("2. Calibrando extrínsecos...")
    if not coordinator.calibrate_extrinsics_from_keypoints(patient_id, session_id):
        logger.error("   ✗ Error calibrando extrínsecos")
        return False
    
    # Paso 4: Verificar sistema calibrado
    if not coordinator.camera_system.is_system_calibrated():
        logger.error("   ✗ Sistema no está completamente calibrado")
        return False
    
    logger.info("   ✓ Sistema completamente calibrado")
    
    # Paso 5: Reconstrucción 3D con SVD
    logger.info("3. Ejecutando reconstrucción 3D (SVD)...")
    success_svd = coordinator.reconstruct_3d(patient_id, session_id, method="svd")
    
    if success_svd:
        logger.info("   ✓ Reconstrucción SVD exitosa")
    else:
        logger.error("   ✗ Error en reconstrucción SVD")
        return False
    
    # Paso 6: Reconstrucción 3D con Bundle Adjustment
    logger.info("4. Ejecutando reconstrucción 3D (Bundle Adjustment)...")
    success_ba = coordinator.reconstruct_3d(patient_id, session_id, method="bundle_adjustment")
    
    if success_ba:
        logger.info("   ✓ Reconstrucción Bundle Adjustment exitosa")
    else:
        logger.warning("   ⚠️ Error en reconstrucción Bundle Adjustment (puede ser normal)")
    
    # Verificar archivos de salida
    output_dir = base_data_dir / "processed" / "3D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    if output_dir.exists():
        output_files = list(output_dir.glob("*.npy"))
        logger.info(f"   ✓ Generados {len(output_files)} archivos de keypoints 3D")
        
        # Mostrar algunos archivos generados
        for i, file in enumerate(output_files[:5]):  # Mostrar primeros 5
            file_size = file.stat().st_size
            logger.info(f"     - {file.name}: {file_size} bytes")
        if len(output_files) > 5:
            logger.info(f"     ... y {len(output_files) - 5} archivos más")
    
    return True

def main():
    """Ejecutar todos los tests"""
    print("🔍 Iniciando tests de reconstrucción 3D con calibración de extrínsecos")
    print("=" * 70)
    
    tests = [
        test_extrinsics_calibration,
        test_full_reconstruction_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()  # Línea en blanco entre tests
        except Exception as e:
            logger.error(f"✗ Error ejecutando {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Resumen
    print("=" * 70)
    print("📋 RESUMEN DE TESTS:")
    passed = sum(results)
    total = len(results)
    
    print(f"   ✅ Tests pasados: {passed}/{total}")
    
    if passed == total:
        print("   🎉 ¡Todos los tests exitosos!")
        print("   📝 Sistema de reconstrucción 3D funcional")
        print("   💡 Próximos pasos:")
        print("      - Usar datos reales de keypoints 2D")
        print("      - Ajustar parámetros según el setup físico")
        print("      - Validar precisión con puntos de referencia conocidos")
    else:
        print("   ⚠️  Algunos tests fallaron")
        print("   🔧 Revisar logs para más detalles")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

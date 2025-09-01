#!/usr/bin/env python3
"""
Script de ejemplo para usar el sistema de reconstrucción 3D
"""

import sys
from pathlib import Path
import logging

# Agregar el directorio del proyecto al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.processing.reconstruction import reconstruct_patient_session, ReconstructionCoordinator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Ejemplo de uso del sistema de reconstrucción 3D"""
    
    # Configuración
    base_data_dir = Path(r"/home/work/Server/data")  # Ajustar según tu configuración
    patient_id = "1"
    session_id = "6"
    camera_ids = [0, 1, 2]  # IDs de las cámaras detectadas
    
    logger.info("=== Ejemplo de Reconstrucción 3D ===")
    
    # Método 1: Usar función directa
    logger.info("1. Reconstrucción con SVD")
    success_svd = reconstruct_patient_session(
        base_data_dir=base_data_dir,
        patient_id=patient_id,
        session_id=session_id,
        camera_ids=camera_ids,
        method="svd",
        use_calibration=True
    )
    
    if success_svd:
        logger.info("✅ Reconstrucción SVD exitosa")
    else:
        logger.error("❌ Falló reconstrucción SVD")
    
    # Método 2: Usar coordinador directamente para más control
    logger.info("2. Reconstrucción con Bundle Adjustment")
    
    coordinator = ReconstructionCoordinator(base_data_dir)
    
    # Configurar parámetros específicos
    coordinator.set_reconstruction_parameters(
        confidence_threshold=0.3,
        min_cameras=2,
        max_reprojection_error=4.0,
        method="bundle_adjustment"
    )
    
    # Inicializar sistema de cámaras
    if coordinator.initialize_camera_system(camera_ids, use_calibration=True):
        logger.info("Sistema de cámaras inicializado")
        
        # Realizar reconstrucción
        success_ba = coordinator.reconstruct_3d(patient_id, session_id, "bundle_adjustment")
        
        if success_ba:
            logger.info("✅ Reconstrucción Bundle Adjustment exitosa")
        else:
            logger.error("❌ Falló reconstrucción Bundle Adjustment")
    else:
        logger.error("❌ Error inicializando sistema de cámaras")
    
    # Método 3: Calibrar cámaras desde imágenes (si es necesario)
    calibration_images_dir = base_data_dir / "calibration_images"
    
    if calibration_images_dir.exists():
        logger.info("3. Calibración de cámaras desde imágenes")
        
        calibration_coordinator = ReconstructionCoordinator(base_data_dir)
        
        if calibration_coordinator.calibrate_cameras(calibration_images_dir, camera_ids):
            logger.info("✅ Calibración exitosa")
            
            # Usar la calibración recién hecha
            success_calibrated = calibration_coordinator.reconstruct_3d(patient_id, session_id, "svd")
            
            if success_calibrated:
                logger.info("✅ Reconstrucción con nueva calibración exitosa")
        else:
            logger.error("❌ Falló calibración")
    else:
        logger.info("3. No se encontraron imágenes de calibración - omitiendo calibración")
    
    # Mostrar sesiones disponibles
    logger.info("4. Sesiones disponibles para reconstrucción:")
    available_sessions = coordinator.get_available_sessions()
    
    for pid, sid in available_sessions:
        logger.info(f"   - Paciente {pid}, Sesión {sid}")
    
    logger.info("=== Ejemplo completado ===")

if __name__ == "__main__":
    main()

"""
Test del sistema de cola multi-worker mejorado
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import time
from pathlib import Path
from backend.processing.coordinator import PoseProcessingCoordinator
from backend.processing.chunk_queue import ChunkProcessingTask

# Configurar logging para testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_worker_system():
    """Test del sistema de worker único"""
    
    print("🧪 Iniciando test del sistema de worker único...")
    
    # Crear coordinator con worker único
    coordinator = PoseProcessingCoordinator(queue_maxsize=10)
    
    # Test 1: Verificar estado inicial
    print("\n📋 Test 1: Estado inicial")
    status = coordinator.get_queue_status()
    print(f"Estado inicial: {status}")
    assert not coordinator.initialized
    assert not coordinator.session_finished
    assert not coordinator.queue_empty_after_session_end
    print("✅ Estado inicial correcto")
    
    # Test 2: Verificar modo secuencial
    print("\n📋 Test 2: Configuración de worker único")
    assert hasattr(coordinator, '_global_detector_lock')
    assert hasattr(coordinator, '_global_video_lock')
    print("✅ Locks globales configurados correctamente")
    
    # Test 3: Señalización de fin de sesión
    print("\n📋 Test 3: Señalización de fin de sesión")
    coordinator.signal_session_end()
    status = coordinator.get_queue_status()
    assert status['session_finished'] == True
    print("✅ Señalización de fin de sesión funciona")
    
    # Test 4: Reset de sesión
    print("\n📋 Test 4: Reset de sesión")
    coordinator.clear_session()
    status = coordinator.get_queue_status()
    assert status['session_finished'] == False
    assert status['queue_empty_after_session_end'] == False
    assert status['processing_mode'] == 'sequential'
    print("✅ Reset de sesión funciona")
    
    # Test 5: Limpieza
    print("\n📋 Test 5: Limpieza")
    coordinator.stop_processing()
    print("✅ Limpieza completada")
    
    print("\n🎉 ¡Todos los tests del worker único pasaron!")

def test_chunk_task():
    """Test de la estructura ChunkProcessingTask"""
    
    print("\n🧪 Test de ChunkProcessingTask...")
    
    task = ChunkProcessingTask(
        video_path=Path("test.mp4"),
        patient_id="patient123",
        session_id="session456", 
        camera_id=0,
        chunk_id="chunk789"
    )
    
    print(f"Tarea creada: {task}")
    assert task.patient_id == "patient123"
    assert task.camera_id == 0
    assert task.chunk_id == "chunk789"
    
    print("✅ ChunkProcessingTask funciona correctamente")

def simulate_session_flow():
    """Simular el flujo completo de una sesión"""
    
    print("\n🎭 Simulando flujo completo de sesión...")
    
    coordinator = PoseProcessingCoordinator()
    
    # Simular chunks agregados (sin inicializar para prueba rápida)
    dummy_path = Path("test_video.mp4")
    
    print("1️⃣ Agregando chunks simulados...")
    for i in range(5):
        success = coordinator.add_chunk_to_queue(
            video_path=dummy_path,
            patient_id="sim_patient",
            session_id="sim_session", 
            camera_id=0,
            chunk_id=str(i)
        )
        print(f"   Chunk {i}: {'❌ Falló (esperado)' if not success else '✅ Agregado'}")
    
    print("2️⃣ Finalizando sesión...")
    coordinator.finish_session()
    
    print("3️⃣ Estado final:")
    status = coordinator.get_queue_status()
    print(f"   Session finished: {status['session_finished']}")
    print(f"   Ready for 3D: {status['ready_for_3d_reconstruction']}")
    print(f"   Processing mode: {status['processing_mode']}")
    
    coordinator.stop_processing()
    print("✅ Simulación completada")

if __name__ == "__main__":
    try:
        test_chunk_task()
        test_single_worker_system()
        simulate_session_flow()
        print("\n🌟 ¡Test completo exitoso!")
        
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()

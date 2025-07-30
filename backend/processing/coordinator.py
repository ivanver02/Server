"""
Coordinador para el procesamiento de pose con múltiples detectores
Sistema de cola thread-safe para procesamiento asíncrono
"""
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Any

from .detectors import VitPoseDetector, MSPNDetector, HRNetDetector, CSPDetector
from .chunk_queue import ChunkProcessingQueue, ChunkProcessingTask

logger = logging.getLogger(__name__)


class PoseProcessingCoordinator:
    """
    Coordinador con cola thread-safe para ejecutar múltiples detectores de pose
    Procesamiento secuencial con locks globales para evitar conflictos
    """
    
    def __init__(self, queue_maxsize: int = 50):
        """
        Inicializar coordinador con todos los detectores disponibles
        
        Args:
            queue_maxsize: Tamaño máximo de la cola de procesamiento
        """
        self.detectors = [
            VitPoseDetector(),
            MSPNDetector(), 
            HRNetDetector(),
            CSPDetector()
        ]
        self.initialized = False
        
        # Sistema de cola
        self.processing_queue = ChunkProcessingQueue(maxsize=queue_maxsize)
        
        # Control de estado de sesión
        self.session_finished = False  # Señal de que no vendrán más chunks
        self.queue_empty_after_session_end = False  # Cola vacía después de finalizar sesión
        
        # Workers y control - SOLO 1 WORKER
        self._worker_thread = None
        self._stop_processing = False
        
        # LOCKS GLOBALES para evitar conflictos
        self._global_detector_lock = threading.Lock()  # Solo 1 detector a la vez
        self._global_video_lock = threading.Lock()     # Solo 1 video a la vez
        self._session_lock = threading.Lock()          # Para cambios de estado de sesión
    
    def initialize_all(self) -> bool:
        """
        Inicializar todos los detectores y arrancar worker único
        
        Returns:
            True si al menos uno se inicializó correctamente
        """
        success_count = 0
        failed_detectors = []
        
        for detector in self.detectors:
            try:
                logger.info(f"🔄 Intentando inicializar detector {detector.model_name}...")
                if detector.initialize():
                    success_count += 1
                    logger.info(f"✅ Detector {detector.model_name} initialized successfully")
                else:
                    failed_detectors.append(detector.model_name)
                    logger.warning(f"⚠️  Failed to initialize detector {detector.model_name}")
            except Exception as e:
                failed_detectors.append(detector.model_name)
                logger.error(f"❌ Error initializing detector {detector.model_name}: {e}")
        
        self.initialized = success_count > 0
        
        if self.initialized:
            # Arrancar UN SOLO worker thread
            self._start_single_worker()
            logger.info(f"🚀 Pose coordinator initialized: {success_count}/{len(self.detectors)} detectors ready")
            if failed_detectors:
                logger.warning(f"⚠️  Failed detectors: {', '.join(failed_detectors)}")
        else:
            logger.error("❌ NO detectors initialized, coordinator FAILED to start")
            logger.error(f"❌ All detectors failed: {', '.join(failed_detectors)}")
        
        return self.initialized
    
    def _start_single_worker(self):
        """Iniciar UN SOLO hilo de procesamiento secuencial"""
        self._stop_processing = False
        
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._process_queue_worker,
                name="ChunkProcessor-Sequential",
                daemon=True
            )
            self._worker_thread.start()
            logger.info("🚀 Worker thread único iniciado para procesamiento secuencial")
    
    def _process_queue_worker(self):
        """
        Worker thread ÚNICO que procesa secuencialmente la cola de chunks
        """
        logger.info("🔄 Worker secuencial iniciando procesamiento...")
        
        while not self._stop_processing:
            try:
                # Obtener próxima tarea (timeout de 2 segundos)
                task = self.processing_queue.get_next_task(timeout=2.0)
                
                if task is None:
                    # Verificar si la sesión terminó y la cola está vacía
                    self._check_session_completion()
                    continue
                
                # Procesar la tarea con LOCK GLOBAL DE VIDEO
                with self._global_video_lock:
                    logger.info(f"🎬 Worker procesando: {task}")
                    self._process_single_chunk(task)
                
            except Exception as e:
                logger.error(f"❌ Error en worker: {e}")
                time.sleep(0.1)  # Breve pausa antes de continuar
        
        logger.info("🛑 Worker secuencial finalizado")
    
    def _check_session_completion(self):
        """
        Verificar si la sesión terminó y la cola está vacía
        """
        with self._session_lock:
            if (self.session_finished and 
                not self.queue_empty_after_session_end and 
                self.processing_queue.is_empty()):
                
                self.queue_empty_after_session_end = True
                stats = self.processing_queue.get_stats()
                
                logger.info("🎯 ¡SESIÓN COMPLETADA! - Cola vacía después de finalizar sesión")
                logger.info("💎 LISTO PARA RECONSTRUCCIÓN 3D - Todos los chunks procesados")
                logger.info(f"📊 Estadísticas finales: {stats['total_processed']} chunks procesados")
    
    def signal_session_end(self):
        """
        Señalar que la sesión ha terminado y no vendrán más chunks
        Esto permite detectar cuándo la cola se vacía por completo
        """
        with self._session_lock:
            self.session_finished = True
            logger.info("� SEÑAL DE FIN DE SESIÓN - No se esperan más chunks")
    
    def _process_single_chunk(self, task: ChunkProcessingTask) -> Dict[str, bool]:
        """
        Procesar un único chunk con todos los detectores SECUENCIALMENTE
        
        Args:
            task: Tarea de procesamiento de chunk
            
        Returns:
            Diccionario con el resultado de cada detector
        """
        results = {}
        start_time = time.time()
        
        logger.info(f"🎬 Procesando chunk {task.chunk_id} para cámara {task.camera_id}")
        
        # PROCESAR CADA DETECTOR SECUENCIALMENTE con LOCK GLOBAL
        for detector in self.detectors:
            if detector.is_initialized:
                with self._global_detector_lock:  # 🔒 LOCK GLOBAL PARA DETECTORES
                    try:
                        logger.info(f"🔄 Procesando con {detector.model_name}...")
                        
                        success = detector.process_chunk(
                            video_path=task.video_path,
                            patient_id=task.patient_id,
                            session_id=task.session_id,
                            camera_id=task.camera_id,
                            chunk_id=task.chunk_id
                        )
                        results[detector.model_name] = success
                        
                        if success:
                            logger.info(f"✅ {detector.model_name} procesó chunk {task.chunk_id} exitosamente")
                        else:
                            logger.warning(f"⚠️  {detector.model_name} falló en chunk {task.chunk_id}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error con {detector.model_name} en chunk {task.chunk_id}: {e}")
                        results[detector.model_name] = False
            else:
                logger.debug(f"⏭️  Saltando {detector.model_name} - no inicializado")
                results[detector.model_name] = False
        
        # Estadísticas de procesamiento
        processing_time = time.time() - start_time
        success_count = sum(results.values())
        
        logger.info(f"🏁 Chunk {task.chunk_id} completado: {success_count}/{len(results)} "
                   f"detectores exitosos ({processing_time:.2f}s)")
        
        return results
    
    
    def add_chunk_to_queue(self, video_path: Path, patient_id: str, session_id: str, 
                          camera_id: int, chunk_id: str) -> bool:
        """
        Agregar un chunk a la cola de procesamiento (método asíncrono)
        
        Args:
            video_path: Ruta al archivo de video del chunk
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_id: ID del chunk
            
        Returns:
            True si se agregó correctamente a la cola
        """
        if not self.initialized:
            logger.error("❌ Coordinator not initialized, cannot add chunk to queue")
            return False
        
        task = ChunkProcessingTask(
            video_path=video_path,
            patient_id=patient_id,
            session_id=session_id,
            camera_id=camera_id,
            chunk_id=chunk_id
        )
        
        success = self.processing_queue.add_task(task)
        
        if success:
            logger.info(f"➕ Chunk {chunk_id} agregado a cola de procesamiento")
        else:
            logger.error(f"❌ No se pudo agregar chunk {chunk_id} a la cola")
        
        return success
    
    def process_chunk(self, video_path: Path, patient_id: str, session_id: str, 
                     camera_id: int, chunk_id: str) -> Dict[str, bool]:
        """
        MÉTODO LEGACY: Procesar un chunk directamente (síncrono)
        
        NOTA: Se mantiene para compatibilidad, pero se recomienda usar add_chunk_to_queue()
        
        Args:
            video_path: Ruta al archivo de video del chunk
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_id: ID del chunk
            
        Returns:
            Diccionario con el resultado de cada detector {detector_name: success}
        """
        logger.warning("⚠️  Usando proceso síncrono legacy. Recomendado: usar add_chunk_to_queue()")
        
        if not self.initialized:
            logger.error("❌ Coordinator not initialized")
            return {}
        
        task = ChunkProcessingTask(
            video_path=video_path,
            patient_id=patient_id,
            session_id=session_id,
            camera_id=camera_id,
            chunk_id=chunk_id
        )
        
        return self._process_single_chunk(task)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de la cola de procesamiento
        
        Returns:
            Diccionario con estadísticas de la cola
        """
        stats = self.processing_queue.get_stats()
        
        with self._session_lock:
            stats.update({
                'session_finished': self.session_finished,
                'queue_empty_after_session_end': self.queue_empty_after_session_end,
                'ready_for_3d_reconstruction': self.queue_empty_after_session_end,
                'worker_alive': self._worker_thread.is_alive() if self._worker_thread else False,
                'processing_mode': 'sequential'  # Indicar modo secuencial
            })
        
        return stats
    
    def stop_processing(self):
        """
        Detener el procesamiento y limpiar recursos
        """
        logger.info("🛑 Deteniendo coordinador de procesamiento...")
        
        self._stop_processing = True
        
        # Esperar a que termine el worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("⚠️  Worker thread no terminó en tiempo esperado")
        
        # Limpiar cola
        self.processing_queue.clear()
        
        # Reiniciar estado
        with self._session_lock:
            self.session_finished = False
            self.queue_empty_after_session_end = False
        
        self.initialized = False
        self._worker_thread = None
        
        logger.info("✅ Coordinador detenido y limpiado")
    
    def clear_session(self):
        """
        Limpiar cola y reiniciar estado para nueva sesión
        Mantiene detectores inicializados y workers corriendo
        """
        logger.info("🧹 Limpiando sesión del coordinador...")
        
        # Limpiar cola
        self.processing_queue.clear()
        
        # Reiniciar estado de sesión
        with self._session_lock:
            self.session_finished = False
            self.queue_empty_after_session_end = False
        
        logger.info("✅ Sesión limpiada, coordinador listo para nueva sesión")
    
    def finish_session(self):
        """
        Señalar que la sesión actual ha terminado
        Esto activará la detección de vaciado completo de cola
        """
        self.signal_session_end()
        logger.info("🏁 Sesión marcada como finalizada - esperando vaciado de cola")

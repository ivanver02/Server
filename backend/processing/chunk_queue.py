"""
Sistema de cola thread-safe para procesamiento de chunks de video
Sistema de análisis de marcha para detección de gonartrosis
"""
import threading
import queue
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ChunkProcessingTask:
    """Estructura de datos para tareas de procesamiento de chunks"""
    video_path: Path
    patient_id: str
    session_id: str
    camera_id: int
    chunk_id: str
    
    def __str__(self):
        return f"ChunkTask(patient={self.patient_id}, session={self.session_id}, camera={self.camera_id}, chunk={self.chunk_id})"


class ChunkProcessingQueue:
    """
    Cola thread-safe para manejar tareas de procesamiento de chunks
    Permite agregar tareas y procesarlas de manera asíncrona
    """
    
    def __init__(self, maxsize: int = 0):
        """
        Inicializar cola de procesamiento
        
        Args:
            maxsize: Tamaño máximo de la cola (0 = ilimitado)
        """
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._processing = False
        self._total_tasks_added = 0
        self._total_tasks_processed = 0
        
        logger.info(f"ChunkProcessingQueue inicializada (maxsize={maxsize})")
    
    def add_task(self, task: ChunkProcessingTask) -> bool:
        """
        Agregar tarea a la cola de procesamiento
        
        Args:
            task: Tarea de procesamiento de chunk
            
        Returns:
            True si se agregó correctamente, False si la cola está llena
        """
        try:
            with self._lock:
                self._queue.put_nowait(task)
                self._total_tasks_added += 1
                
            logger.info(f"✅ Tarea agregada a cola: {task}")
            logger.debug(f"Cola actual: {self._queue.qsize()} tareas pendientes")
            return True
            
        except queue.Full:
            logger.error(f"❌ Cola llena, no se pudo agregar tarea: {task}")
            return False
        except Exception as e:
            logger.error(f"❌ Error agregando tarea a cola: {e}")
            return False
    
    def get_next_task(self, timeout: Optional[float] = None) -> Optional[ChunkProcessingTask]:
        """
        Obtener próxima tarea de la cola
        
        Args:
            timeout: Tiempo máximo de espera (None = esperar indefinidamente)
            
        Returns:
            ChunkProcessingTask si hay una disponible, None si timeout
        """
        try:
            with self._lock:
                task = self._queue.get(timeout=timeout)
                self._total_tasks_processed += 1
                
            logger.info(f"🔄 Tarea extraída de cola: {task}")
            logger.debug(f"Cola actual: {self._queue.qsize()} tareas pendientes")
            return task
            
        except queue.Empty:
            # Timeout o cola vacía
            return None
        except Exception as e:
            logger.error(f"❌ Error extrayendo tarea de cola: {e}")
            return None
    
    def is_empty(self) -> bool:
        """
        Verificar si la cola está vacía
        
        Returns:
            True si la cola está vacía
        """
        with self._lock:
            return self._queue.empty()
    
    def size(self) -> int:
        """
        Obtener tamaño actual de la cola
        
        Returns:
            Número de tareas pendientes en la cola
        """
        with self._lock:
            return self._queue.qsize()
    
    def get_stats(self) -> dict:
        """
        Obtener estadísticas de la cola
        
        Returns:
            Diccionario con estadísticas de la cola
        """
        with self._lock:
            return {
                'current_size': self._queue.qsize(),
                'total_added': self._total_tasks_added,
                'total_processed': self._total_tasks_processed,
                'pending': self._total_tasks_added - self._total_tasks_processed,
                'is_processing': self._processing
            }
    
    def set_processing_status(self, processing: bool):
        """
        Establecer estado de procesamiento
        
        Args:
            processing: True si se está procesando, False si no
        """
        with self._lock:
            self._processing = processing
    
    def clear(self):
        """
        Limpiar toda la cola (útil para cancelar sesión)
        """
        with self._lock:
            cleared_count = self._queue.qsize()
            
            # Vaciar cola
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            
            # Reiniciar contadores
            self._total_tasks_added = 0
            self._total_tasks_processed = 0
            self._processing = False
            
        logger.info(f"🧹 Cola limpiada: {cleared_count} tareas eliminadas")

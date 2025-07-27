"""
Sincronizador de videos para extracción de frames sincronizados
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import logging
import time

from ..data import VideoInfo, SyncFrame, SyncConfig

logger = logging.getLogger(__name__)


class VideoSynchronizer:
    """
    Sincronizador de múltiples videos para extracción de frames exactamente
    en los mismos timestamps, manteniendo la sincronización temporal
    """
    
    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig()
        self.video_captures: Dict[int, cv2.VideoCapture] = {}
        self.video_infos: Dict[int, VideoInfo] = {}
        self.sync_fps: Optional[float] = None
        self.is_initialized = False
        self.max_duration = 0.0
        self.min_duration = float('inf')
    
    def add_video(self, camera_id: int, video_path: Path) -> bool:
        """
        Añadir video para sincronización
        
        Args:
            camera_id: ID único de la cámara
            video_path: Ruta al archivo de video
            
        Returns:
            True si se añadió correctamente
        """
        try:
            if not video_path.exists():
                logger.error(f"Video no encontrado: {video_path}")
                return False
            
            # Abrir video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"No se puede abrir video: {video_path}")
                return False
            
            # Obtener información del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Crear info del video
            video_info = VideoInfo(
                camera_id=camera_id,
                video_path=video_path,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                width=width,
                height=height
            )
            
            # Almacenar
            self.video_captures[camera_id] = cap
            self.video_infos[camera_id] = video_info
            
            # Actualizar duraciones
            self.max_duration = max(self.max_duration, duration)
            self.min_duration = min(self.min_duration, duration)
            
            logger.info(f"Video añadido - Cámara {camera_id}: {fps:.1f}fps, "
                       f"{total_frames} frames, {duration:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error añadiendo video cámara {camera_id}: {e}")
            return False
    
    def initialize_sync(self) -> bool:
        """
        Inicializar sincronización determinando FPS común
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            if len(self.video_infos) < 1:
                logger.error("Se necesita al menos 1 video para sincronización")
                return False
            
            # Determinar FPS de sincronización
            if self.config.target_fps is None:
                # Usar el menor FPS para asegurar que todos los videos puedan seguir el ritmo
                self.sync_fps = min(info.fps for info in self.video_infos.values())
            else:
                self.sync_fps = self.config.target_fps
                
                # Verificar que todos los videos pueden soportar este FPS
                max_possible_fps = min(info.fps for info in self.video_infos.values())
                if self.sync_fps > max_possible_fps:
                    logger.warning(f"FPS objetivo {self.sync_fps} reducido a {max_possible_fps}")
                    self.sync_fps = max_possible_fps
            
            logger.info(f"Sincronización inicializada: {self.sync_fps:.1f} FPS, "
                       f"{len(self.video_infos)} cámaras")
            
            # Log información de cada video
            for camera_id, info in self.video_infos.items():
                logger.info(f"  Cámara {camera_id}: {info.fps:.1f}fps, "
                          f"{info.total_frames} frames, {info.duration:.1f}s")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando sincronización: {e}")
            return False
    
    def get_sync_frame_at_timestamp(self, timestamp: float) -> Optional[SyncFrame]:
        """
        Obtener frame sincronizado en un timestamp específico
        
        Args:
            timestamp: Timestamp en segundos
            
        Returns:
            Frame sincronizado o None si hay error
        """
        if not self.is_initialized:
            logger.error("Sincronizador no inicializado")
            return None
        
        # Verificar que el timestamp esté en rango
        if timestamp < 0 or timestamp > self.min_duration:
            return None
        
        try:
            camera_frames = {}
            available_cameras = []
            sync_quality_scores = []
            
            for camera_id, cap in self.video_captures.items():
                info = self.video_infos[camera_id]
                
                # Calcular número de frame para este timestamp
                target_frame = timestamp * info.fps
                frame_number = int(round(target_frame))
                
                # Verificar que el frame existe
                if frame_number >= info.total_frames:
                    continue
                
                # Calcular precisión de sincronización
                actual_timestamp = frame_number / info.fps
                sync_error = abs(timestamp - actual_timestamp)
                sync_quality = max(0, 1 - (sync_error / self.config.sync_tolerance))
                
                if sync_quality < self.config.quality_threshold:
                    logger.warning(f"Calidad de sync baja para cámara {camera_id}: {sync_quality:.2f}")
                
                # Posicionar video en el frame correcto
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    camera_frames[camera_id] = frame.copy()
                    available_cameras.append(camera_id)
                    sync_quality_scores.append(sync_quality)
            
            if not camera_frames:
                return None
            
            # Calcular calidad de sincronización general
            overall_sync_quality = np.mean(sync_quality_scores) if sync_quality_scores else 0.0
            
            # Calcular número de frame sincronizado basado en FPS de sync
            sync_frame_number = int(timestamp * self.sync_fps)
            
            return SyncFrame(
                timestamp=timestamp,
                frame_number=sync_frame_number,
                camera_frames=camera_frames,
                available_cameras=available_cameras,
                sync_quality=overall_sync_quality
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo frame sincronizado en {timestamp:.3f}s: {e}")
            return None
    
    def iterate_synchronized_frames(self) -> Iterator[SyncFrame]:
        """
        Iterar sobre todos los frames sincronizados según la configuración
        
        Yields:
            Frames sincronizados
        """
        if not self.is_initialized:
            logger.error("Sincronizador no inicializado")
            return
        
        # Determinar rango de tiempo
        start_time = self.config.start_time
        end_time = self.config.end_time or self.min_duration
        end_time = min(end_time, self.min_duration)
        
        if start_time >= end_time:
            logger.error(f"Rango de tiempo inválido: {start_time} >= {end_time}")
            return
        
        # Calcular intervalo de tiempo entre frames
        frame_interval_time = self.config.frame_interval / self.sync_fps
        
        logger.info(f"Iterando frames sincronizados: {start_time:.1f}s - {end_time:.1f}s, "
                   f"intervalo {self.config.frame_interval} frames ({frame_interval_time:.3f}s)")
        
        current_time = start_time
        frame_count = 0
        
        while current_time <= end_time:
            sync_frame = self.get_sync_frame_at_timestamp(current_time)
            
            if sync_frame and len(sync_frame.available_cameras) > 0:
                frame_count += 1
                yield sync_frame
                
                if frame_count % 100 == 0:  # Log progreso cada 100 frames
                    logger.info(f"Procesados {frame_count} frames sincronizados...")
            
            current_time += frame_interval_time
        
        logger.info(f"Extracción completada: {frame_count} frames sincronizados")
    
    def get_timestamps_list(self, frame_interval: Optional[int] = None) -> List[float]:
        """
        Obtener lista de timestamps para procesamiento
        
        Args:
            frame_interval: Intervalo entre frames (None = usar config)
            
        Returns:
            Lista de timestamps en segundos
        """
        if not self.is_initialized:
            return []
        
        interval = frame_interval or self.config.frame_interval
        frame_interval_time = interval / self.sync_fps
        
        start_time = self.config.start_time
        end_time = self.config.end_time or self.min_duration
        end_time = min(end_time, self.min_duration)
        
        timestamps = []
        current_time = start_time
        
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += frame_interval_time
        
        return timestamps
    
    def get_sync_info(self) -> Dict[str, any]:
        """Obtener información de sincronización"""
        if not self.is_initialized:
            return {}
        
        return {
            'sync_fps': self.sync_fps,
            'num_cameras': len(self.video_infos),
            'min_duration': self.min_duration,
            'max_duration': self.max_duration,
            'config': {
                'target_fps': self.config.target_fps,
                'frame_interval': self.config.frame_interval,
                'start_time': self.config.start_time,
                'end_time': self.config.end_time,
                'sync_tolerance': self.config.sync_tolerance,
                'quality_threshold': self.config.quality_threshold
            },
            'camera_infos': {
                camera_id: {
                    'fps': info.fps,
                    'total_frames': info.total_frames,
                    'duration': info.duration,
                    'resolution': f"{info.width}x{info.height}",
                    'video_path': str(info.video_path)
                }
                for camera_id, info in self.video_infos.items()
            }
        }
    
    def cleanup(self):
        """Limpiar recursos"""
        try:
            for cap in self.video_captures.values():
                if cap.isOpened():
                    cap.release()
            
            self.video_captures.clear()
            self.video_infos.clear()
            self.is_initialized = False
            self.max_duration = 0.0
            self.min_duration = float('inf')
            
            logger.info("VideoSynchronizer limpiado")
            
        except Exception as e:
            logger.error(f"Error limpiando VideoSynchronizer: {e}")


def create_synchronizer_from_videos(video_paths: Dict[int, Path], 
                                   config: Optional[SyncConfig] = None) -> VideoSynchronizer:
    """
    Crear sincronizador a partir de videos de múltiples cámaras
    
    Args:
        video_paths: Diccionario {camera_id: video_path}
        config: Configuración de sincronización
        
    Returns:
        VideoSynchronizer configurado
    """
    synchronizer = VideoSynchronizer(config)
    
    logger.info(f"Creando sincronizador para {len(video_paths)} videos")
    
    # Añadir todos los videos
    success_count = 0
    for camera_id, video_path in video_paths.items():
        if synchronizer.add_video(camera_id, video_path):
            success_count += 1
        else:
            logger.warning(f"No se pudo añadir video cámara {camera_id}: {video_path}")
    
    logger.info(f"Videos añadidos al sincronizador: {success_count}/{len(video_paths)}")
    
    return synchronizer

"""
Configuración general del sistema de procesamiento de video
Sistema de análisis de marcha para detección de gonartrosis
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent

@dataclass
class ServerConfig:
    """Configuración del servidor Flask"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = True
    max_content_length: int = 100 * 1024 * 1024  # 100MB
    upload_folder: str = str(BASE_DIR / "data" / "unprocessed")


@dataclass
class DataConfig:
    """Configuración de directorios de datos"""
    base_data_dir: Path = BASE_DIR / "data"
    unprocessed_dir: Path = base_data_dir / "unprocessed"
    processed_dir: Path = base_data_dir / "processed"
    
    # Directorios específicos de datos procesados
    keypoints_2d_dir: Path = processed_dir / "2D_keypoints"
    keypoints_3d_dir: Path = processed_dir / "3D_keypoints"
    photos_dir: Path = processed_dir / "photos_from_video"
    
    # Directorio de logs
    logs_dir: Path = BASE_DIR / "logs"
    
    # Directorio para videos anotados
    annotated_videos_dir: Path = processed_dir / "annotated_videos"
    
    # Extensiones de archivos permitidas
    video_extensions: List[str] = field(default_factory=lambda: ['.mp4', '.avi', '.mov', '.mkv'])
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])
    
    def ensure_directories(self):
        """Crear todos los directorios necesarios"""
        dirs_to_create = [
            self.base_data_dir,
            self.unprocessed_dir,
            self.processed_dir,
            self.keypoints_2d_dir,
            self.keypoints_3d_dir,
            self.photos_dir,
            self.logs_dir,
            self.annotated_videos_dir
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class MMPoseConfig:
    """Configuración para modelos MMPose"""
    models_dir: Path = BASE_DIR / "mmpose_models"
    configs_dir: Path = models_dir / "configs"
    checkpoints_dir: Path = models_dir / "checkpoints"
    
    # Configuraciones específicas de cada detector
    vitpose: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py',
        'pose2d_weights': 'checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth',
        'device': 'cuda:0'
    })
    
    mspn: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/td-hm_4xmspn50_8xb32-210e_coco-256x192.py',
        'pose2d_weights': 'checkpoints/4xmspn50_coco_256x192-7b837afb_20201123.pth',
        'device': 'cuda:0'
    })
    
    hrnet: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
        'pose2d_weights': 'checkpoints/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
        'device': 'cuda:0'
    })
    
    csp: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/td-hm_cspnet-l_8xb64-210e_coco-256x192.py',
        'pose2d_weights': 'checkpoints/cspnet-l_coco_256x192-1b50f8dc_20201123.pth',
        'device': 'cuda:0'
    })
    
    def ensure_directories(self):
        """Crear directorios para modelos"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ProcessingConfig:
    """Configuración para procesamiento de video"""
    # Configuración de extracción de frames
    target_fps: int = 15
    max_frames_per_chunk: int = 100
    
    # Configuración de procesamiento paralelo
    max_workers: int = 4
    enable_parallel_processing: bool = True
    
    # Configuración de videos anotados
    save_annotated_videos: bool = False  # Si guardar videos con keypoints dibujados
    
    # Configuración de modelos por defecto
    default_models: List[str] = field(default_factory=lambda: ['hrnet_w48_coco', 'vitpose_huge_coco'])
    coco_models: List[str] = field(default_factory=lambda: ['hrnet_w48_coco', 'vitpose_huge_coco'])
    extended_models: List[str] = field(default_factory=lambda: [])


@dataclass 
class SynchronizationConfig:
    """Configuración para sincronización de videos"""
    # Configuración de sincronización temporal
    target_fps: int = 15
    frame_interval: int = 1  # Procesar cada N frames
    sync_tolerance: float = 0.1  # Tolerancia de sincronización en segundos
    quality_threshold: float = 0.8  # Umbral de calidad mínima
    
    # Configuración de extracción de frames
    max_frame_diff: int = 5  # Máxima diferencia de frames entre cámaras
    enable_frame_interpolation: bool = False  # Interpolación de frames faltantes
    
    # Configuración de validación
    min_sync_duration: float = 1.0  # Duración mínima para considerar sincronización válida
    max_sync_duration: float = 30.0  # Duración máxima de procesamiento

@dataclass
class ReconstructionConfig:
    """Configuración para reconstrucción 3D"""
    # Configuración de calibración
    min_cameras_for_triangulation: int = 2
    max_reprojection_error: float = 2.0
    
    # Configuración de triangulación
    min_confidence_score: float = 0.3
    use_opencv_triangulation: bool = True
    
    # Configuración de suavizado
    enable_temporal_smoothing: bool = True
    smoothing_window_size: int = 5

# Instancias globales de configuración
server_config = ServerConfig()
processing_config = ProcessingConfig()
synchronization_config = SynchronizationConfig()
reconstruction_config = ReconstructionConfig()
data_config = DataConfig()
mmpose_config = MMPoseConfig()

# Inicializar directorios al importar
data_config.ensure_directories()
mmpose_config.ensure_directories()

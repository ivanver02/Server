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
            self.logs_dir
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
    hrnet_w48: Dict[str, str] = field(default_factory=lambda: {
        'model_name': 'hrnet-w48_coco_256x192',
        'config': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
        'checkpoint': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
        'device': 'auto'  # auto detecta CUDA o CPU
    })
    
    vitpose: Dict[str, str] = field(default_factory=lambda: {
        'model_name': 'vitpose_huge_coco',
        'config': 'vitpose-h-multi-coco',
        'checkpoint': None,  # MMPose descargará automáticamente
        'device': 'auto'
    })
    
    rtmpose: Dict[str, str] = field(default_factory=lambda: {
        'model_name': 'rtmpose_l_coco_256x192',
        'config': 'rtmpose-l_8xb256-420e_coco-256x192',
        'checkpoint': None,  # MMPose descargará automáticamente
        'device': 'auto'
    })
    
    wholebody: Dict[str, str] = field(default_factory=lambda: {
        'model_name': 'wholebody_coco_133',
        'config': 'td-hm_hrnet-w48_8xb32-210e_wholebody-384x288',
        'checkpoint': None,  # MMPose descargará automáticamente
        'device': 'auto'
    })
    
    # Configuraciones de modelos legacy (mantener por compatibilidad)
    model_configs: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.model_configs is None:
            self.model_configs = {
                'hrnet_w48_coco': {
                    'config': 'configs/pose2d/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py',
                    'checkpoint': 'checkpoints/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.pth',
                    'keypoints': 17,
                    'gpu': 'cuda:0'
                },
                'vitpose_huge_coco': {
                    'config': 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py',
                    'checkpoint': 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth',
                    'keypoints': 17,
                    'gpu': 'cuda:0'
                },
                'resnet50_rle_coco': {
                    'config': 'configs/pose2d/td-hm_res50_rle-8xb64-210e_coco-256x192.py',
                    'checkpoint': 'checkpoints/td-hm_res50_rle-8xb64-210e_coco-256x192.pth',
                    'keypoints': 17,
                    'gpu': 'cuda:1'
                },
                'wholebody_coco': {
                    'config': 'configs/pose2d/wholebody_2d_keypoint_topdown_coco-wholebody.py',
                    'checkpoint': 'checkpoints/wholebody_2d_keypoint_topdown_coco-wholebody.pth',
                    'keypoints': 133,  # 17 body + 6 feet + 42 hands + 68 face
                    'gpu': 'cuda:1'
                }
            }
    
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

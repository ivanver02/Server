"""
Configuración general del sistema de procesamiento de video
Sistema de análisis de marcha para detección de gonartrosis
"""
import os
from dataclasses import dataclass
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
class ProcessingConfig:
    """Configuración del procesamiento de video"""
    # FPS para extracción de frames
    target_fps: int = 15
    
    # Modelos MMPose a usar
    coco_models: List[str] = None
    extended_models: List[str] = None
    
    # GPU configuration
    primary_gpu: str = 'cuda:0'
    secondary_gpu: str = 'cuda:1'
    
    # Thresholds
    confidence_threshold: float = 0.3
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.coco_models is None:
            self.coco_models = [
                'hrnet_w48_coco',
                'hrnet_w32_coco'
            ]
        
        if self.extended_models is None:
            self.extended_models = [
                'resnet50_rle_coco',
                'wholebody_coco'
            ]
            
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'hrnet_w48_coco': 0.6,
                'hrnet_w32_coco': 0.4,
                'resnet50_rle_coco': 1.0,
                'wholebody_coco': 1.0
            }

@dataclass
class ReconstructionConfig:
    """Configuración para reconstrucción 3D"""
    # Método de triangulación
    triangulation_method: str = 'dlt'  # Direct Linear Transform
    
    # Optimización
    optimize_triangulation: bool = True
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    
    # Validación
    max_reprojection_error: float = 5.0  # píxeles
    min_cameras_for_point: int = 2

@dataclass
class DataConfig:
    """Configuración de directorios de datos"""
    base_data_dir: Path = BASE_DIR / "data"
    unprocessed_dir: Path = base_data_dir / "unprocessed"
    processed_dir: Path = base_data_dir / "processed"
    
    # Subdirectorios procesados
    photos_dir: Path = processed_dir / "photos_from_video"
    keypoints_2d_dir: Path = processed_dir / "2D_keypoints"
    keypoints_3d_dir: Path = processed_dir / "3D_keypoints"
    
    # Logs
    logs_dir: Path = BASE_DIR / "logs"
    
    def ensure_directories(self):
        """Crear todos los directorios necesarios"""
        dirs_to_create = [
            self.base_data_dir,
            self.unprocessed_dir,
            self.processed_dir,
            self.photos_dir,
            self.keypoints_2d_dir,
            self.keypoints_3d_dir,
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
    
    # Configuraciones de modelos
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

# Instancias globales de configuración
server_config = ServerConfig()
processing_config = ProcessingConfig()
reconstruction_config = ReconstructionConfig()
data_config = DataConfig()
mmpose_config = MMPoseConfig()

# Inicializar directorios al importar
data_config.ensure_directories()
mmpose_config.ensure_directories()

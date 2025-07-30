# Inicialización del módulo config
from .settings import (
    server_config,
    gpu_config,
    processing_config,
    data_config,
    mmpose_config
)

__all__ = [
    # Settings
    'server_config',
    'gpu_config',
    'processing_config',
    'data_config',
    'mmpose_config',
]

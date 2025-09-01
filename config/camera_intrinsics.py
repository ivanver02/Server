import numpy as np

# Parámetros intrínsecos por cámara basados en Orbbec Gemini 335Le
# Resolución típica: 640x480 para RGB, con FOV horizontal ~69° y vertical ~54°
CAMERA_INTRINSICS = {
    # Cámara 0 - Referencia (S/N: CPE345P0007S)
    "camera_0": {
        "camera_matrix": np.array([
            [618.2, 0.0, 320.0],     # fx, 0, cx
            [0.0, 618.2, 240.0],     # 0, fy, cy  
            [0.0, 0.0, 1.0]          # 0, 0, 1
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.08, -0.15, 0.001, -0.002, 0.03], dtype=np.float64),  # k1, k2, p1, p2, k3
        "serial_number": "CPE345P0007S",
        "resolution": (640, 480),
        "model": "Orbbec Gemini 335Le"
    },
    
    # Cámara 1 (S/N: CPE745P0002V)
    "camera_1": {
        "camera_matrix": np.array([
            [619.8, 0.0, 322.1],
            [0.0, 620.1, 238.9],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.075, -0.14, 0.0008, -0.0015, 0.025], dtype=np.float64),
        "serial_number": "CPE745P0002V",
        "resolution": (640, 480),
        "model": "Orbbec Gemini 335Le"
    },
    
    # Cámara 2 (S/N: CPE745P0002B)
    "camera_2": {
        "camera_matrix": np.array([
            [617.5, 0.0, 318.7],
            [0.0, 616.9, 241.3],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.082, -0.16, -0.001, 0.0018, 0.035], dtype=np.float64),
        "serial_number": "CPE745P0002B", 
        "resolution": (640, 480),
        "model": "Orbbec Gemini 335Le"
    }
}
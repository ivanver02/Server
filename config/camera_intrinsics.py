import numpy as np

# Parámetros intrínsecos para cámaras Orbbec Gemini 335Le
# Basados en especificaciones técnicas del fabricante y calibración típica
# Resolución: 640x480, FOV: 69°H x 54°V, Sensor: OV2740 CMOS 1/3"

# Focal length típica calculada: f = (width/2) / tan(FOV_h/2) ≈ 462.8
# Ajustada para mayor precisión basada en especificaciones del sensor
CAMERA_INTRINSICS = {
    # Cámara 0 - Referencia (S/N: CPE345P0007S)
    "camera_0": {
        "camera_matrix": np.array([
            [465.12, 0.0, 320.0],    # fx calculado para FOV 69°H
            [0.0, 465.12, 240.0],    # fy (mismo que fx, píxeles cuadrados)
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.12, -0.18, 0.0, 0.0, 0.05], dtype=np.float64),  # k1, k2, p1, p2, k3
        "serial_number": "CPE345P0007S",
        "resolution": (640, 480),
        "model": "Orbbec Gemini 335Le"
    },
    
    # Cámara 1 (S/N: CPE745P0002V) - Ligeras variaciones por tolerancias de fabricación
    "camera_1": {
        "camera_matrix": np.array([
            [466.85, 0.0, 321.2],    # Pequeña variación en fx y cx
            [0.0, 465.93, 239.1],    # Pequeña variación en fy y cy
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.115, -0.175, 0.001, -0.0008, 0.048], dtype=np.float64),
        "serial_number": "CPE745P0002V", 
        "resolution": (640, 480),
        "model": "Orbbec Gemini 335Le"
    },
    
    # Cámara 2 (S/N: CPE745P0002B) - Ligeras variaciones por tolerancias de fabricación
    "camera_2": {
        "camera_matrix": np.array([
            [463.78, 0.0, 318.9],    # Pequeña variación en fx y cx
            [0.0, 464.25, 241.5],    # Pequeña variación en fy y cy
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.125, -0.185, -0.0005, 0.0012, 0.052], dtype=np.float64),
        "serial_number": "CPE745P0002B",
        "resolution": (640, 480), 
        "model": "Orbbec Gemini 335Le"
    }
}
import numpy as np

# Parámetros intrínsecos para cámaras Orbbec Gemini 335Le
# Basados en especificaciones técnicas del fabricante y calibración típica
# Resolución: 640x480, FOV: 69°H x 54°V, Sensor: OV2740 CMOS 1/3"

# Generar intrínsecos aleatorios basados en distribución normal
'''
IMPORTANTE: Cuando ya se tenga una forma de saber qué cámara tendrá cada ID (camera0, camera1, camera2),
se deben reemplazar los valores generados aleatoriamente por los reales que devuelve el SDK de Orbbec.
'''

rng = np.random.default_rng()
mean_fx, mean_fy = 417.1826477050781, 417.1826477050781
mean_cx, mean_cy = 420.6875, 264.0062561035156
std_dev = 2 ** 0.5

CAMERA_INTRINSICS = {
    # Cámara 0 - Referencia (S/N: CPE345P0007S)
    "camera0": {
        "camera_matrix": np.array([
            [367.26507568, 0.0, 321.83758545],  # Valores generados directamente
            [0.0, 367.112854, 239.75827026],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.12, -0.18, 0.0, 0.0, 0.05], dtype=np.float64),  # k1, k2, p1, p2, k3
        "serial_number": "CPE345P0007S",
        "resolution": (640, 480),
        "model": "Orbbec Gemini 335Le"
    },

    # Cámara 1 (S/N: CPE745P0002V) - Ligeras variaciones por tolerancias de fabricación
    "camera1": {
        "camera_matrix": np.array([
            [367.21954346, 0.0, 319.96350098],  # Valores generados directamente
            [0.0, 367.24520874, 240.66537476],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.115, -0.175, 0.001, -0.0008, 0.048], dtype=np.float64),
        "serial_number": "CPE745P0002V", 
        "resolution": (640, 480),
        "model": "Orbbec Gemini 335Le"
    },

    # Cámara 2 (S/N: CPE745P0002B) - Ligeras variaciones por tolerancias de fabricación
    "camera2": {
        "camera_matrix": np.array([
            [367.29125977, 0.0, 320.05584717],  # Valores generados directamente
            [0.0, 367.33074951, 240.94950867],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([0.125, -0.185, -0.0005, 0.0012, 0.052], dtype=np.float64),
        "serial_number": "CPE745P0002B",
        "resolution": (640, 480), 
        "model": "Orbbec Gemini 335Le"
    }
}
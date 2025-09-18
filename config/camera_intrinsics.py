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
        "distortion_coeffs": np.array([-3.31241898e-02,  3.54048684e-02, -4.07491325e-06, -2.58066721e-04,
       -1.23792645e-02], dtype=np.float64),  # k1, k2, p1, p2, k3
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
        "distortion_coeffs": np.array([-0.03170585,  0.03431713,  0.00032519, -0.00052007, -0.01199829], dtype=np.float64),
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
        "distortion_coeffs": np.array([-0.0318352 ,  0.03448541,  0.00026838, -0.0003188 , -0.0119219], dtype=np.float64),
        "serial_number": "CPE745P0002B",
        "resolution": (640, 480), 
        "model": "Orbbec Gemini 335Le"
    },

    # Cámara 3 (S/N: CPE745P00018) - Ligeras variaciones por tolerancias de fabricación
    "camera3": {
        "camera_matrix": np.array([
            [366.95544434, 0.0, 319.40429688],  # Valores generados directamente
            [0.0, 366.93487549, 238.6277771],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([-0.031112,  0.033254,  0.000365, -0.000915, -0.011411], dtype=np.float64),
        "serial_number": "CPE745P00018",
        "resolution": (640, 480), 
        "model": "Orbbec Gemini 335Le"
    },

    # Cámara 4 (S/N: CPE345P0007P) - Ligeras variaciones por tolerancias de fabricación
    "camera4": {
        "camera_matrix": np.array([
            [366.72042847, 0.0, 322.06997681],  # Valores generados directamente
            [0.0, 366.63381958, 240.39222717],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        "distortion_coeffs": np.array([-0.031542,  0.034544,  0.000525, -0.000181, -0.011977], dtype=np.float64),
        "serial_number": "CPE345P0007P",
        "resolution": (640, 480), 
        "model": "Orbbec Gemini 335Le"
    }
}
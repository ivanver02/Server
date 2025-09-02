"""Prueba sintética de reconstrucción 3D.

Genera:
 - Puntos 3D arbitrarios (simulan esqueleto sencillo)
 - 3 cámaras con intrínsecos y extrínsecos conocidos
 - Proyecciones 2D (formato similar a processed: (K,2) + (K,) confidence)

Luego:
 - Triangula con SVD
 - Refinamiento con Bundle Adjustment
 - Calcula error de reproyección (px) y error 3D respecto al ground truth

Objetivo: verificar que con datos limpios el pipeline produce error casi nulo.
"""
from __future__ import annotations

import numpy as np
from camera import Camera
from triangulation_svd import triangulate_frame_svd
from triangulation_bundle_adjustment import refine_frame_bundle_adjustment
from reprojection import reprojection_error


def generate_ground_truth_3d() -> np.ndarray:
    # 17 puntos estilo COCO simplificado (arbitrarios en torno a z≈2m)
    pts = np.array([
        [0.0, 1.70, 2.5],   # Nose
        [-0.15, 1.55, 2.5], # Eye L
        [0.15, 1.55, 2.5],  # Eye R
        [-0.25, 1.35, 2.5], # Shoulder L
        [0.25, 1.35, 2.5],  # Shoulder R
        [-0.30, 1.00, 2.5], # Elbow L
        [0.30, 1.00, 2.5],  # Elbow R
        [-0.35, 0.70, 2.55],# Wrist L
        [0.35, 0.70, 2.55], # Wrist R
        [0.0, 1.10, 2.5],   # Mid torso
        [-0.15, 0.90, 2.55],# Hip L
        [0.15, 0.90, 2.55], # Hip R
        [-0.15, 0.50, 2.65],# Knee L
        [0.15, 0.50, 2.65], # Knee R
        [-0.15, 0.10, 2.75],# Ankle L
        [0.15, 0.10, 2.75], # Ankle R
        [0.0, 0.00, 2.80],  # Foot mid
    ], dtype=np.float64)
    return pts


def create_cameras():
    # Intrínsecos base (ligeras variaciones)
    fx0 = fy0 = 470.0; cx0, cy0 = 320.0, 240.0
    fx1 = 472.0; fy1 = 469.0; cx1, cy1 = 321.5, 239.0
    fx2 = 468.5; fy2 = 471.0; cx2, cy2 = 318.0, 241.2

    cams = {}
    for cid in ["camera0", "camera1", "camera2"]:
        cams[cid] = Camera.create(cid)  # Inicializa K/dists desde config si existen; luego sobreescribimos K

    cams["camera0"].K = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]], dtype=np.float64)
    cams["camera1"].K = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]], dtype=np.float64)
    cams["camera2"].K = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]], dtype=np.float64)

    # Extrínsecos: camera0 referencia
    cams["camera0"].R = np.eye(3)
    cams["camera0"].t = np.zeros((3, 1))

    # Camera1: desplazada 0.5m a la derecha y rotada -8° yaw
    yaw1 = np.deg2rad(-8)
    R1 = np.array([
        [np.cos(yaw1), 0, np.sin(yaw1)],
        [0, 1, 0],
        [-np.sin(yaw1), 0, np.cos(yaw1)],
    ])
    t1 = np.array([[0.5], [0.0], [0.0]])
    cams["camera1"].R = R1
    cams["camera1"].t = t1

    # Camera2: desplazada 0.6m a la izquierda y rotada +10° yaw
    yaw2 = np.deg2rad(10)
    R2 = np.array([
        [np.cos(yaw2), 0, np.sin(yaw2)],
        [0, 1, 0],
        [-np.sin(yaw2), 0, np.cos(yaw2)],
    ])
    t2 = np.array([[-0.6], [0.0], [0.05]])
    cams["camera2"].R = R2
    cams["camera2"].t = t2

    return cams


def project_points(cams, pts3d, noise_px=0.0):
    frame_keypoints = {}
    for cid, cam in cams.items():
        pts2d = cam.project(pts3d)
        if noise_px > 0:
            pts2d += np.random.normal(0, noise_px, pts2d.shape)
        conf = np.ones((pts2d.shape[0],), dtype=np.float64)
        frame_keypoints[cid] = (pts2d, conf)
    return frame_keypoints


def compute_3d_error(gt, pred):
    valid = ~np.isnan(pred[:, 0])
    if not np.any(valid):
        return np.nan
    diff = gt[valid] - pred[valid]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def main():
    np.random.seed(42)
    pts3d_gt = generate_ground_truth_3d()
    cams = create_cameras()
    frame = project_points(cams, pts3d_gt, noise_px=0.0)

    print("Triangulación SVD...")
    pts3d_svd = triangulate_frame_svd(cams, frame)
    err3d_svd = compute_3d_error(pts3d_gt, pts3d_svd)
    reproj_svd = reprojection_error(pts3d_svd, cams, frame)
    print(f"Error 3D medio SVD: {err3d_svd:.6f} m")
    for cid, e in reproj_svd.items():
        print(f"  Reproj {cid}: {e:.6f} px")

    print("\nRefinamiento Bundle Adjustment...")
    pts3d_ba = refine_frame_bundle_adjustment(pts3d_svd, cams, frame)
    err3d_ba = compute_3d_error(pts3d_gt, pts3d_ba)
    reproj_ba = reprojection_error(pts3d_ba, cams, frame)
    print(f"Error 3D medio BA:  {err3d_ba:.6f} m")
    for cid, e in reproj_ba.items():
        print(f"  Reproj {cid}: {e:.6f} px")

    # Diferencia entre SVD y BA
    if not np.isnan(err3d_svd) and not np.isnan(err3d_ba):
        print(f"\nMejora BA (Δ error 3D): {err3d_svd - err3d_ba:.6e} m")

    # Comprobación rápida de que los errores de reproyección son ~0
    max_reproj_svd = np.nanmax(list(reproj_svd.values()))
    max_reproj_ba = np.nanmax(list(reproj_ba.values()))
    print(f"Max reproj SVD: {max_reproj_svd:.6e} px | Max reproj BA: {max_reproj_ba:.6e} px")


if __name__ == "__main__":
    main()
"""
Prueba sintética del sistema de reconstrucción 3D.
Genera datos sintéticos con geometría conocida para verificar que el pipeline funciona correctamente.
"""

import numpy as np
import logging
from backend.processing.reconstruction.camera import Camera
from backend.processing.reconstruction.triangulation_svd import triangulate_svd
from backend.processing.reconstruction.triangulation_bundle_adjustment import triangulate_bundle_adjustment
from backend.processing.reconstruction.reprojection import reproject_and_validate

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_scene():
    """
    Genera una escena sintética con geometría conocida.
    """
    # Definir puntos 3D sintéticos (en metros)
    # Simular una persona de pie a ~2 metros de la cámara
    points_3d_true = np.array([
        [0.0, 1.8, 2.0],   # Cabeza
        [0.0, 1.6, 2.0],   # Cuello
        [-0.2, 1.4, 2.0],  # Hombro izquierdo
        [0.2, 1.4, 2.0],   # Hombro derecho
        [-0.3, 1.0, 2.0],  # Codo izquierdo
        [0.3, 1.0, 2.0],   # Codo derecho
        [-0.1, 0.8, 2.0],  # Muñeca izquierda
        [0.1, 0.8, 2.0],   # Muñeca derecha
        [0.0, 1.2, 2.0],   # Centro torso
        [-0.1, 0.6, 2.0],  # Cadera izquierda
        [0.1, 0.6, 2.0],   # Cadera derecha
        [-0.1, 0.2, 2.0],  # Rodilla izquierda
        [0.1, 0.2, 2.0],   # Rodilla derecha
        [-0.1, -0.2, 2.0], # Tobillo izquierdo
        [0.1, -0.2, 2.0],  # Tobillo derecho
    ])
    
    logger.info(f"Puntos 3D verdaderos generados: {len(points_3d_true)} puntos")
    logger.info(f"Rango X: [{points_3d_true[:, 0].min():.2f}, {points_3d_true[:, 0].max():.2f}]")
    logger.info(f"Rango Y: [{points_3d_true[:, 1].min():.2f}, {points_3d_true[:, 1].max():.2f}]")
    logger.info(f"Rango Z: [{points_3d_true[:, 2].min():.2f}, {points_3d_true[:, 2].max():.2f}]")
    
    return points_3d_true

def generate_synthetic_cameras():
    """
    Genera configuración de cámaras sintéticas.
    """
    cameras = {}
    
    # Parámetros intrínsecos sintéticos (similares a Orbbec)
    K_base = np.array([
        [460.0, 0.0, 320.0],
        [0.0, 460.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Camera 0 - Referencia (origen del mundo)
    cameras["camera0"] = {
        "K": K_base.copy(),
        "R": np.eye(3),
        "t": np.zeros((3, 1))
    }
    
    # Camera 1 - A la derecha y ligeramente rotada
    R1 = np.array([
        [0.9659, 0.0, 0.2588],
        [0.0, 1.0, 0.0],
        [-0.2588, 0.0, 0.9659]
    ])  # Rotación ~15 grados en Y
    t1 = np.array([[0.5], [0.0], [0.1]])  # 50cm a la derecha, 10cm adelante
    
    cameras["camera1"] = {
        "K": K_base + np.array([[2.0, 0, 1.0], [0, 1.5, -1.0], [0, 0, 0]]),  # Pequeñas variaciones
        "R": R1,
        "t": t1
    }
    
    # Camera 2 - A la izquierda y ligeramente rotada
    R2 = np.array([
        [0.9659, 0.0, -0.2588],
        [0.0, 1.0, 0.0],
        [0.2588, 0.0, 0.9659]
    ])  # Rotación ~-15 grados en Y
    t2 = np.array([[-0.6], [0.1], [0.05]])  # 60cm a la izquierda, 10cm arriba, 5cm adelante
    
    cameras["camera2"] = {
        "K": K_base + np.array([[-1.5, 0, -0.8], [0, -1.2, 1.5], [0, 0, 0]]),  # Pequeñas variaciones
        "R": R2,
        "t": t2
    }
    
    logger.info("Cámaras sintéticas generadas:")
    for cam_id, cam_data in cameras.items():
        logger.info(f"  {cam_id}: t={cam_data['t'].flatten()}, det(R)={np.linalg.det(cam_data['R']):.6f}")
    
    return cameras

def project_points_synthetic(points_3d, camera_params):
    """
    Proyecta puntos 3D a 2D usando parámetros de cámara conocidos.
    """
    K, R, t = camera_params["K"], camera_params["R"], camera_params["t"]
    
    # Transformar a coordenadas de cámara
    points_cam = R @ points_3d.T + t
    
    # Proyectar a imagen
    points_2d_homo = K @ points_cam
    points_2d = points_2d_homo[:2] / points_2d_homo[2]
    
    return points_2d.T

def test_synthetic_reconstruction():
    """
    Prueba completa con datos sintéticos.
    """
    print("=== PRUEBA SINTÉTICA RECONSTRUCCIÓN 3D ===\n")
    
    # 1. Generar escena sintética
    print("1. Generando escena sintética...")
    points_3d_true = generate_synthetic_scene()
    
    # 2. Generar cámaras sintéticas
    print("\n2. Generando cámaras sintéticas...")
    camera_params = generate_synthetic_cameras()
    
    # 3. Proyectar puntos 3D a 2D (simulando detección de keypoints)
    print("\n3. Proyectando puntos 3D a 2D...")
    keypoints_2d = {}
    
    for cam_id, cam_data in camera_params.items():
        points_2d = project_points_synthetic(points_3d_true, cam_data)
        
        # Añadir pequeño ruido gaussiano para simular imprecisión de detección
        noise = np.random.normal(0, 0.5, points_2d.shape)  # 0.5px de ruido
        points_2d_noisy = points_2d + noise
        
        keypoints_2d[cam_id] = points_2d_noisy
        
        logger.info(f"  {cam_id}: {len(points_2d)} puntos proyectados")
        logger.info(f"    Rango X: [{points_2d[:, 0].min():.1f}, {points_2d[:, 0].max():.1f}] px")
        logger.info(f"    Rango Y: [{points_2d[:, 1].min():.1f}, {points_2d[:, 1].max():.1f}] px")
    
    # 4. Crear objetos Camera con los parámetros sintéticos
    print("\n4. Configurando objetos Camera...")
    cameras = {}
    
    for cam_id, cam_data in camera_params.items():
        # Crear camera temporal solo para almacenar parámetros
        camera = Camera("camera0")  # Dummy ID
        camera.camera_id = cam_id
        camera.K = cam_data["K"].copy()
        camera.R = cam_data["R"].copy()
        camera.t = cam_data["t"].copy()
        
        cameras[cam_id] = camera
        logger.info(f"  {cam_id}: fx={camera.K[0,0]:.1f}, fy={camera.K[1,1]:.1f}")
    
    # 5. Reconstrucción con SVD
    print("\n5. Reconstrucción con SVD...")
    points_3d_svd = triangulate_svd(cameras, keypoints_2d)
    
    # 6. Reconstrucción con Bundle Adjustment
    print("\n6. Reconstrucción con Bundle Adjustment...")
    points_3d_ba = triangulate_bundle_adjustment(cameras, keypoints_2d, points_3d_svd)
    
    # 7. Validación por reproyección
    print("\n7. Validación por reproyección...")
    validation_svd = reproject_and_validate(cameras, points_3d_svd, keypoints_2d)
    validation_ba = reproject_and_validate(cameras, points_3d_ba, keypoints_2d)
    
    # 8. Análisis de errores
    print("\n8. Análisis de errores...")
    
    # Error de reconstrucción 3D
    error_3d_svd = np.linalg.norm(points_3d_svd - points_3d_true, axis=1)
    error_3d_ba = np.linalg.norm(points_3d_ba - points_3d_true, axis=1)
    
    print(f"\n--- RESULTADOS ---")
    print(f"Error 3D SVD:")
    print(f"  - Medio: {np.mean(error_3d_svd):.6f} m")
    print(f"  - Máximo: {np.max(error_3d_svd):.6f} m")
    
    print(f"\nError 3D Bundle Adjustment:")
    print(f"  - Medio: {np.mean(error_3d_ba):.6f} m")
    print(f"  - Máximo: {np.max(error_3d_ba):.6f} m")
    
    # Error de reproyección
    svd_reproj_error = validation_svd.get("global", {}).get("mean_error", np.nan)
    ba_reproj_error = validation_ba.get("global", {}).get("mean_error", np.nan)
    
    print(f"\nError reproyección:")
    print(f"  - SVD: {svd_reproj_error:.3f} px")
    print(f"  - Bundle Adjustment: {ba_reproj_error:.3f} px")
    
    # Verificación punto por punto para el primer punto
    print(f"\n--- VERIFICACIÓN PRIMER PUNTO ---")
    print(f"Punto 3D verdadero:    {points_3d_true[0]}")
    print(f"Punto 3D SVD:          {points_3d_svd[0]}")
    print(f"Punto 3D Bundle Adj:   {points_3d_ba[0]}")
    print(f"Error 3D SVD:          {error_3d_svd[0]:.6f} m")
    print(f"Error 3D Bundle Adj:   {error_3d_ba[0]:.6f} m")
    
    for cam_id in cameras.keys():
        original_2d = keypoints_2d[cam_id][0]
        reproj_2d = cameras[cam_id].project_points(points_3d_svd[0].reshape(1, 3))[0]
        error_px = np.linalg.norm(reproj_2d - original_2d)
        print(f"{cam_id}: Original {original_2d} -> Reproj {reproj_2d} -> Error {error_px:.3f}px")
    
    print(f"\n=== FIN PRUEBA SINTÉTICA ===")
    
    return {
        "error_3d_svd": np.mean(error_3d_svd),
        "error_3d_ba": np.mean(error_3d_ba),
        "error_reproj_svd": svd_reproj_error,
        "error_reproj_ba": ba_reproj_error
    }

if __name__ == "__main__":
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    results = test_synthetic_reconstruction()
    
    # Verificar que los errores son aceptables
    print(f"\n--- VERIFICACIÓN FINAL ---")
    if results["error_3d_svd"] < 0.001:  # < 1mm
        print("✓ Error 3D SVD: ACEPTABLE")
    else:
        print("✗ Error 3D SVD: DEMASIADO ALTO")
    
    if results["error_reproj_svd"] < 1.0:  # < 1px
        print("✓ Error reproyección SVD: ACEPTABLE")
    else:
        print("✗ Error reproyección SVD: DEMASIADO ALTO")
    
    if results["error_3d_ba"] < 0.001:  # < 1mm
        print("✓ Error 3D Bundle Adjustment: ACEPTABLE")
    else:
        print("✗ Error 3D Bundle Adjustment: DEMASIADO ALTO")
    
    if results["error_reproj_ba"] < 1.0:  # < 1px
        print("✓ Error reproyección Bundle Adjustment: ACEPTABLE")
    else:
        print("✗ Error reproyección Bundle Adjustment: DEMASIADO ALTO")

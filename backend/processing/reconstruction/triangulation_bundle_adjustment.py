"""Refinamiento por Bundle Adjustment (optimiza sólo puntos 3D por frame)."""
import numpy as np
from typing import Dict, Tuple
from scipy.optimize import least_squares
from camera import Camera


def _residuals(points_3d_flat: np.ndarray, cameras: Dict[str, Camera], frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]], valid_mask: np.ndarray) -> np.ndarray:
    K = valid_mask.shape[0]
    pts3d = points_3d_flat.reshape(K, 3)
    res = []
    for cid, cam in cameras.items():
        coords, conf = frame_keypoints[cid]
        valid = valid_mask & (conf > 0)
        if not np.any(valid):
            continue
        proj = cam.project(pts3d[valid])
        diff = (proj - coords[valid])  # (N,2)
        res.append(diff.reshape(-1))
    if not res:
        return np.zeros(0)
    return np.concatenate(res)


def refine_frame_bundle_adjustment(initial_points_3d: np.ndarray, cameras: Dict[str, Camera], frame_keypoints: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    pts3d = initial_points_3d.copy()
    valid_mask = ~np.isnan(pts3d[:, 0])
    if not np.any(valid_mask):
        return pts3d
    
    # Filtrar puntos con valores extremos que pueden causar problemas
    valid_points = pts3d[valid_mask]
    reasonable_mask = (
        (np.abs(valid_points[:, 0]) < 10.0) &  # X entre -10 y 10 metros
        (np.abs(valid_points[:, 1]) < 5.0) &   # Y entre -5 y 5 metros  
        (valid_points[:, 2] > 0.1) &           # Z positivo (delante de cámara)
        (valid_points[:, 2] < 20.0)            # Z menor a 20 metros
    )
    
    if not np.any(reasonable_mask):
        print("Warning: No hay puntos 3D en rango razonable para Bundle Adjustment")
        return pts3d
    
    # Crear máscara combinada
    full_reasonable_mask = valid_mask.copy()
    full_reasonable_mask[valid_mask] = reasonable_mask
    
    if np.sum(full_reasonable_mask) < 5:
        print(f"Warning: Solo {np.sum(full_reasonable_mask)} puntos válidos para BA")
        return pts3d
    
    try:
        x0 = pts3d.reshape(-1)  # Usar todos los puntos (23*3 = 69)
        
        # Verificar que no hay valores problemáticos
        if not np.all(np.isfinite(x0)):
            print(f"Warning: Hay valores NaN/Inf en puntos iniciales: {np.sum(~np.isfinite(x0))} de {len(x0)}")
            # Reemplazar NaN/Inf con valores razonables
            x0 = np.where(np.isfinite(x0), x0, 0.0)
        
        print(f"Optimizando {np.sum(full_reasonable_mask)} de {len(pts3d)} puntos 3D")
        valid_pts = pts3d[full_reasonable_mask]
        print(f"Rango inicial - X:[{valid_pts[:, 0].min():.3f}, {valid_pts[:, 0].max():.3f}], Y:[{valid_pts[:, 1].min():.3f}, {valid_pts[:, 1].max():.3f}], Z:[{valid_pts[:, 2].min():.3f}, {valid_pts[:, 2].max():.3f}]")
        print(f"x0 stats: min={x0.min():.3f}, max={x0.max():.3f}, finite={np.sum(np.isfinite(x0))}/{len(x0)}")
        
        result = least_squares(
            _residuals, x0, 
            args=(cameras, frame_keypoints, full_reasonable_mask),  # Usar máscara más restrictiva
            method="lm",  # Levenberg-Marquardt
            max_nfev=200,
            ftol=1e-8,
            xtol=1e-8
        )
        
        if result.success:
            print(f"Bundle Adjustment exitoso. Costo: {result.cost:.6f}")
            refined = result.x.reshape(-1, 3)  # Ahora sí es (23, 3)
            # Mantener NaN en puntos no triangulados originalmente
            refined[~valid_mask] = np.nan
            return refined
        else:
            print(f"Warning: Bundle Adjustment no convergió ({result.message})")
            return pts3d
        
    except Exception as e:
        print(f"Warning: Bundle Adjustment falló ({e}), usando puntos SVD")
        return pts3d
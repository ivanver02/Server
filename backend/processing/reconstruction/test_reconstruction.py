"""Prueba completa de reconstrucción 3D (chunk 0) para patient1 session8.

No genera datos nuevos: asume que existen keypoints 2D procesados en:
data/processed/2D_keypoints/patient1/session8/cameraX/(coordinates|confidence)/{frame}_0.npy

Pasos:
1. Calcular extrínsecos (camera0 referencia)
2. Para cada frame: triangulación SVD -> guardar 3D
3. Reproyección y métricas
4. Refinar con Bundle Adjustment y comparar
"""
import os
from pathlib import Path
import numpy as np
from calculate_extrinsics import calculate_extrinsics
from triangulation_svd import triangulate_frame_svd
from triangulation_bundle_adjustment import refine_frame_bundle_adjustment
from reprojection import reprojection_error

PATIENT_ID = "patient1"
SESSION_ID = "session8"
CHUNK_ID = 0

# Raíz del proyecto (Server/) y rutas absolutas para evitar problemas al ejecutar desde distintos cwd
_ROOT = Path(__file__).resolve().parents[3]
BASE_2D_DIR = str(_ROOT / "data" / "processed" / "2D_keypoints")
BASE_3D_DIR = str(_ROOT / "data" / "processed" / "3D_keypoints")


def _list_cameras(session_path: str):
    return sorted([d for d in os.listdir(session_path) if d.startswith("camera") and os.path.isdir(os.path.join(session_path, d))])


def _frame_ids(camera_dir: str, chunk_id: int):
    coord_dir = os.path.join(camera_dir, "coordinates")
    if not os.path.isdir(coord_dir):
        return []
    ids = []
    suf = f"_{chunk_id}.npy"
    for f in os.listdir(coord_dir):
        if f.endswith(suf):
            try:
                ids.append(int(f.split("_")[0]))
            except ValueError:
                pass
    return sorted(ids)


def _load_frame(camera_path: str, frame_id: int, chunk_id: int):
    c_path = os.path.join(camera_path, "coordinates", f"{frame_id}_{chunk_id}.npy")
    conf_path = os.path.join(camera_path, "confidence", f"{frame_id}_{chunk_id}.npy")
    coords = np.load(c_path)
    conf = np.load(conf_path)
    return coords, conf


def main():
    session_path = os.path.join(BASE_2D_DIR, PATIENT_ID, SESSION_ID)
    # Debug de ruta
    print(f"Buscando sesión en: {session_path}")
    if not os.path.isdir(session_path):
        print("No existe la ruta de sesión, abortando.")
        return
    cameras_ids = _list_cameras(session_path)
    if not cameras_ids:
        print("No se detectaron cámaras.")
        return
    if "camera0" not in cameras_ids:
        print("Se requiere camera0.")
        return
    frame_ids = _frame_ids(os.path.join(session_path, "camera0"), CHUNK_ID)
    if not frame_ids:
        print("No se encontraron frames para chunk 0.")
        return
    print(f"Cámaras detectadas: {cameras_ids}")
    print(f"Total frames chunk 0: {len(frame_ids)} (frame_ids={frame_ids[0]}..{frame_ids[-1]})")

    # 1. Extrínsecos
    print("Calculando extrínsecos...")
    cameras = calculate_extrinsics(PATIENT_ID, SESSION_ID, CHUNK_ID, BASE_2D_DIR)
    for cid, cam in cameras.items():
        print(f"{cid}: R=\n{cam.R}\n t={cam.t.ravel()}")

    # Preparar salida 3D
    out_session_dir = os.path.join(BASE_3D_DIR, PATIENT_ID, SESSION_ID)
    os.makedirs(out_session_dir, exist_ok=True)

    svd_errors = []
    ba_errors = []

    for fid in frame_ids:
        frame_data = {}
        for cid in cameras_ids:
            cdir = os.path.join(session_path, cid)
            try:
                frame_data[cid] = _load_frame(cdir, fid, CHUNK_ID)
            except FileNotFoundError:
                # Si falta en alguna cámara, se sigue con las restantes
                continue
        if len(frame_data) < 2:
            continue
        # 2. Triangulación SVD
        points_3d_svd = triangulate_frame_svd(cameras, frame_data)
        # Guardar (una sola versión; podría sobrescribirse luego por BA si se desea)
        out_file = os.path.join(out_session_dir, f"{fid}_{CHUNK_ID}.npy")
        np.save(out_file, points_3d_svd)
        # 3. Reproyección SVD
        err_svd = reprojection_error(points_3d_svd, cameras, frame_data)
        mean_err_svd = np.nanmean(list(err_svd.values())) if err_svd else np.nan
        svd_errors.append(mean_err_svd)
        # 4. Bundle Adjustment
        points_3d_ba = refine_frame_bundle_adjustment(points_3d_svd, cameras, frame_data)
        err_ba = reprojection_error(points_3d_ba, cameras, frame_data)
        mean_err_ba = np.nanmean(list(err_ba.values())) if err_ba else np.nan
        ba_errors.append(mean_err_ba)
        # Mostrar resumen por algunos frames
        if fid == frame_ids[0] or fid == frame_ids[-1] or fid % 50 == 0:
            print(f"Frame {fid}: mean reproj SVD={mean_err_svd:.3f}px BA={mean_err_ba:.3f}px")

    if svd_errors:
        print(f"Resumen SVD: media={np.nanmean(svd_errors):.3f}px")
    if ba_errors:
        print(f"Resumen BA:  media={np.nanmean(ba_errors):.3f}px")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script para generar video anotado con keypoints del ensemble
Procesa el chunk 0 del paciente 1, sesión 6, cámara 0
"""

import cv2
import numpy as np
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ensemble_keypoints(base_data_dir: Path, patient_id: str, session_id: str, camera_id: int, chunk_number: int):
    """
    Cargar keypoints del ensemble para un chunk específico
    Retorna: dict {frame_number: {'coordinates': array(N,2), 'confidence': array(N)}}
    """
    keypoints_dir = base_data_dir / "processed" / "2D_keypoints" / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
    coordinates_dir = keypoints_dir / "coordinates"
    confidence_dir = keypoints_dir / "confidence"
    
    if not coordinates_dir.exists() or not confidence_dir.exists():
        logger.error(f"No se encontraron directorios de keypoints: {coordinates_dir}")
        return {}
    
    # Buscar archivos del chunk específico
    chunk_files = list(coordinates_dir.glob(f"*_{chunk_number}.npy"))
    
    if not chunk_files:
        logger.error(f"No se encontraron archivos de keypoints para chunk {chunk_number}")
        return {}
    
    keypoints = {}
    
    for coord_file in chunk_files:
        try:
            # Extraer número de frame del nombre del archivo (formato: frame_number_chunk_number.npy)
            frame_number = int(coord_file.stem.split('_')[0])
            
            # Cargar coordenadas y confianza
            coordinates = np.load(coord_file)
            confidence_file = confidence_dir / coord_file.name
            
            if confidence_file.exists():
                confidence = np.load(confidence_file)
                keypoints[frame_number] = {
                    'coordinates': coordinates,
                    'confidence': confidence
                }
                logger.debug(f"Cargado frame {frame_number}: {coordinates.shape[0]} keypoints")
            else:
                logger.warning(f"No se encontró archivo de confianza para frame {frame_number}")
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Error procesando archivo {coord_file}: {e}")
            continue
    
    logger.info(f"Cargados keypoints de {len(keypoints)} frames para chunk {chunk_number}")
    return keypoints

"""Esqueleto COCO estándar (17 keypoints).
Índices asumidos (COCO):
0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear 5:left_shoulder 6:right_shoulder
7:left_elbow 8:right_elbow 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
"""
# Lista de aristas COCO comúnmente usada
SKELETON_EDGES = [
    (5,7), (7,9),        # Brazo izquierdo
    (6,8), (8,10),       # Brazo derecho
    (5,6),               # Hombros
    (5,11), (6,12),      # Hombro a cadera
    (11,12),             # Caderas
    (11,13), (13,15),    # Pierna izquierda
    (12,14), (14,16),    # Pierna derecha
    (0,5), (0,6),        # Nariz a hombros
    (0,1), (0,2),        # Nariz a ojos
    (1,3), (2,4)         # Ojo a oreja
]

def draw_keypoints_on_frame(frame, coordinates, confidence, confidence_threshold=0.1, draw_skeleton=True):
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    n = len(coordinates)

    # Dibujar esqueleto primero (para que los puntos queden encima)
    if draw_skeleton:
        for a, b in SKELETON_EDGES:
            if a < n and b < n:
                conf_a = confidence[a] if a < len(confidence) else 0
                conf_b = confidence[b] if b < len(confidence) else 0
                if conf_a > confidence_threshold and conf_b > confidence_threshold:
                    xa, ya = int(coordinates[a,0]), int(coordinates[a,1])
                    xb, yb = int(coordinates[b,0]), int(coordinates[b,1])
                    if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
                        conf_line = (conf_a + conf_b) * 0.5
                        c_int = min(255, int(conf_line * 255))
                        color = (0, c_int, 255 - c_int)
                        thickness = 2 if conf_line > 0.5 else 1
                        cv2.line(annotated_frame, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)

    # Dibujar puntos
    for i, (coord, conf) in enumerate(zip(coordinates, confidence)):
        if conf > confidence_threshold:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < w and 0 <= y < h:
                color_intensity = min(255, int(conf * 255))
                color = (0, color_intensity, 255 - color_intensity)
                radius = 4 if conf > 0.6 else 3
                cv2.circle(annotated_frame, (x, y), radius, color, -1, cv2.LINE_AA)
                cv2.putText(annotated_frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return annotated_frame

def create_annotated_video(base_data_dir: Path, patient_id: str, session_id: str, camera_id: int, chunk_number: int,
                           draw_skeleton: bool = True, confidence_threshold: float = 0.1):
    """
    Crear video anotado con keypoints del ensemble
    """
    # Rutas de archivos
    video_path = base_data_dir / "unprocessed" / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}" / f"{chunk_number}.mp4"
    
    if not video_path.exists():
        logger.error(f"No se encontró el video: {video_path}")
        return False
    
    # Cargar keypoints del ensemble
    keypoints = load_ensemble_keypoints(base_data_dir, patient_id, session_id, camera_id, chunk_number)
    
    if not keypoints:
        logger.error("No se pudieron cargar los keypoints del ensemble")
        return False
    
    # Crear directorio de salida
    output_dir = base_data_dir / "processed" / "test_annotated" / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{chunk_number}.mp4"
    
    # Abrir video de entrada
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"No se pudo abrir el video: {video_path}")
        return False
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video original: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Configurar writer de video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"No se pudo crear el video de salida: {output_path}")
        cap.release()
        return False
    
    # Procesar frame por frame
    frame_number = 0
    frames_processed = 0
    frames_with_keypoints = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Buscar keypoints para este frame
        if frame_number in keypoints:
            frame_keypoints = keypoints[frame_number]
            annotated_frame = draw_keypoints_on_frame(
                frame,
                frame_keypoints['coordinates'],
                frame_keypoints['confidence'],
                confidence_threshold=confidence_threshold,
                draw_skeleton=draw_skeleton,
            )
            frames_with_keypoints += 1
            
        else:
            # Si no hay keypoints, usar frame original con mensaje
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"Frame: {frame_number} | No keypoints", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Escribir frame anotado
        out.write(annotated_frame)
        frames_processed += 1
        
        if frames_processed % 30 == 0:  # Log cada segundo aproximadamente
            logger.info(f"Procesados {frames_processed}/{total_frames} frames")
        
        frame_number += 1
    
    # Limpiar recursos
    cap.release()
    out.release()
    
    logger.info(f"Video anotado creado exitosamente: {output_path}")
    logger.info(f"Frames procesados: {frames_processed}")
    logger.info(f"Frames con keypoints: {frames_with_keypoints}")
    
    return True

def process_videos(patient_id: str, session_id: str, cameras_count: int, chunk_number: int = 0, draw_skeleton: bool = True, confidence_threshold: float = 0.1):
    """
    Función principal para generar videos anotados para un chunk específico de todas las cámaras.
    """
    base_data_dir = Path(r"/home/work/Server/data")  # Ajustar ruta según tu configuración

    if not base_data_dir.exists():
        logger.error(f"No se encontró el directorio base: {base_data_dir}")
        logger.error("Ajusta la variable 'base_data_dir' en el script según tu configuración")
        return

    logger.info(f"Iniciando procesamiento de video para paciente {patient_id}, sesión {session_id}, chunk {chunk_number}")

    for camera_id in range(cameras_count):
        logger.info(f"Procesando cámara {camera_id}...")
        
        success = create_annotated_video(
            base_data_dir, patient_id, session_id, camera_id, chunk_number,
            draw_skeleton=draw_skeleton, confidence_threshold=confidence_threshold
        )

        if success:
            logger.info(f"Procesamiento completado exitosamente para la cámara {camera_id}")
        else:
            logger.error(f"Error en el procesamiento para la cámara {camera_id}")

if __name__ == "__main__":
    # Ejemplo de uso: procesar paciente 1, sesión 8, para 4 cámaras, chunk 0.
    process_videos(patient_id="1", session_id="8", cameras_count=4, chunk_number=0)

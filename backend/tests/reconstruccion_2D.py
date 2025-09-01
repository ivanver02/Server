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

def draw_keypoints_on_frame(frame, coordinates, confidence, confidence_threshold=0.1):
    """
    Dibujar keypoints en un frame
    """
    annotated_frame = frame.copy()
    
    for i, (coord, conf) in enumerate(zip(coordinates, confidence)):
        if conf > confidence_threshold:  # Solo dibujar keypoints con confianza suficiente
            x, y = int(coord[0]), int(coord[1])
            
            # Verificar que las coordenadas están dentro del frame
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Color basado en la confianza (rojo para alta confianza, azul para baja)
                color_intensity = min(255, int(conf * 255))
                color = (0, color_intensity, 255 - color_intensity)  # BGR
                
                # Dibujar círculo para el keypoint
                cv2.circle(annotated_frame, (x, y), 3, color, -1)
                
                # Opcional: dibujar número del keypoint
                cv2.putText(annotated_frame, str(i), (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return annotated_frame

def create_annotated_video(base_data_dir: Path, patient_id: str, session_id: str, camera_id: int, chunk_number: int):
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
    output_path = output_dir / f"chunk_{chunk_number}_annotated.mp4"
    
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
                frame_keypoints['confidence']
            )
            frames_with_keypoints += 1
            
            # Agregar información de frame en la esquina
            info_text = f"Frame: {frame_number} | Keypoints: {len(frame_keypoints['coordinates'])}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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

def main():
    """Función principal"""
    # Configuración - ajustar según tu estructura
    base_data_dir = Path(r"C:\Users\Juan Cantizani\Server\data")  # Ajustar ruta según tu configuración
    patient_id = "1"
    session_id = "6"
    camera_id = 0
    chunk_number = 0
    
    logger.info(f"Iniciando procesamiento para paciente {patient_id}, sesión {session_id}, cámara {camera_id}, chunk {chunk_number}")
    
    # Verificar que existe el directorio base
    if not base_data_dir.exists():
        logger.error(f"No se encontró el directorio base: {base_data_dir}")
        logger.error("Ajusta la variable 'base_data_dir' en el script según tu configuración")
        return
    
    # Crear video anotado
    success = create_annotated_video(base_data_dir, patient_id, session_id, camera_id, chunk_number)
    
    if success:
        logger.info("✅ Procesamiento completado exitosamente")
    else:
        logger.error("❌ Error en el procesamiento")

if __name__ == "__main__":
    main()

"""
Script de Test Simple para Videos de Cámaras
===========================================

Script simplificado para probar inicialmente el procesamiento de videos
sin dependencias complejas, verificando que los videos se pueden leer
y extraer frames básicos.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import json
import time
from typing import Dict, List
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de rutas
TEST_BASE_PATH = Path(__file__).parent
VIDEO_PATHS = {
    'camera0': TEST_BASE_PATH / 'camera0' / '0.mp4',
    'camera1': TEST_BASE_PATH / 'camera1' / '0.mp4', 
    'camera2': TEST_BASE_PATH / 'camera2' / '0.mp4'
}
PROCESSED_DATA_PATH = TEST_BASE_PATH / 'processed_data'
FRAMES_PATH = PROCESSED_DATA_PATH / 'frames'

def analyze_video(video_path: Path, camera_id: str) -> Dict:
    """
    Analizar un video y extraer información básica
    
    Args:
        video_path: Ruta al archivo de video
        camera_id: Identificador de la cámara
        
    Returns:
        Diccionario con información del video
    """
    logger.info(f"📹 Analizando video: {camera_id}")
    
    info = {
        'camera_id': camera_id,
        'file_path': str(video_path),
        'exists': video_path.exists(),
        'file_size_mb': 0,
        'duration_seconds': 0,
        'fps': 0,
        'total_frames': 0,
        'width': 0,
        'height': 0,
        'readable': False,
        'sample_frames_extracted': 0
    }
    
    if not video_path.exists():
        logger.error(f"❌ {camera_id}: Archivo no encontrado - {video_path}")
        return info
    
    # Tamaño del archivo
    info['file_size_mb'] = round(video_path.stat().st_size / (1024 * 1024), 2)
    
    try:
        # Abrir video con OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"❌ {camera_id}: No se puede abrir con OpenCV")
            return info
        
        info['readable'] = True
        
        # Obtener propiedades del video
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calcular duración
        if info['fps'] > 0:
            info['duration_seconds'] = round(info['total_frames'] / info['fps'], 2)
        
        # Extraer algunos frames de muestra
        sample_frames = extract_sample_frames(cap, camera_id, max_frames=5)
        info['sample_frames_extracted'] = len(sample_frames)
        
        cap.release()
        
        logger.info(f"✅ {camera_id}: {info['width']}x{info['height']}, {info['fps']:.1f} FPS, {info['duration_seconds']}s")
        
    except Exception as e:
        logger.error(f"❌ {camera_id}: Error procesando video - {e}")
    
    return info

def extract_sample_frames(cap, camera_id: str, max_frames: int = 5) -> List[str]:
    """
    Extraer frames de muestra del video
    
    Args:
        cap: Objeto VideoCapture de OpenCV
        camera_id: ID de la cámara
        max_frames: Máximo número de frames a extraer
        
    Returns:
        Lista de rutas de frames guardados
    """
    # Crear directorio para frames
    frames_dir = FRAMES_PATH / camera_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return extracted_frames
        
        # Calcular posiciones de frames equidistantes
        if total_frames <= max_frames:
            frame_positions = list(range(total_frames))
        else:
            step = total_frames // max_frames
            frame_positions = [i * step for i in range(max_frames)]
        
        for i, frame_pos in enumerate(frame_positions):
            # Ir a la posición del frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Leer frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Guardar frame
                frame_filename = f"sample_frame_{i:02d}.jpg"
                frame_path = frames_dir / frame_filename
                
                success = cv2.imwrite(str(frame_path), frame)
                
                if success:
                    extracted_frames.append(str(frame_path))
                    logger.debug(f"  Frame {i+1}/{len(frame_positions)} guardado: {frame_filename}")
                else:
                    logger.warning(f"  ⚠️ No se pudo guardar frame {i+1}")
            else:
                logger.warning(f"  ⚠️ No se pudo leer frame en posición {frame_pos}")
    
    except Exception as e:
        logger.error(f"Error extrayendo frames de muestra: {e}")
    
    return extracted_frames

def test_opencv_installation():
    """Verificar que OpenCV está instalado y funcionando"""
    try:
        logger.info("🔧 Verificando instalación de OpenCV...")
        
        # Verificar versión
        cv_version = cv2.__version__
        logger.info(f"✅ OpenCV versión: {cv_version}")
        
        # Crear imagen de prueba
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :] = [100, 150, 200]  # Color de prueba
        
        # Guardar imagen de prueba
        test_path = PROCESSED_DATA_PATH / "opencv_test.jpg"
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(test_path), test_img)
        
        if success and test_path.exists():
            logger.info("✅ OpenCV puede crear y guardar imágenes")
            test_path.unlink()  # Eliminar archivo de prueba
            return True
        else:
            logger.error("❌ OpenCV no puede guardar imágenes")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error verificando OpenCV: {e}")
        return False

def generate_test_report(video_info_list: List[Dict]):
    """
    Generar reporte de análisis de videos
    
    Args:
        video_info_list: Lista de información de videos analizados
    """
    report_path = PROCESSED_DATA_PATH / "video_analysis_report.json"
    
    # Calcular estadísticas generales
    total_videos = len(video_info_list)
    readable_videos = sum(1 for info in video_info_list if info['readable'])
    total_duration = sum(info['duration_seconds'] for info in video_info_list)
    total_frames = sum(info['total_frames'] for info in video_info_list)
    total_size_mb = sum(info['file_size_mb'] for info in video_info_list)
    
    report = {
        'timestamp': time.time(),
        'summary': {
            'total_videos': total_videos,
            'readable_videos': readable_videos,
            'total_duration_seconds': round(total_duration, 2),
            'total_frames': total_frames,
            'total_size_mb': round(total_size_mb, 2),
            'average_fps': round(sum(info['fps'] for info in video_info_list if info['fps'] > 0) / max(1, readable_videos), 2)
        },
        'videos': video_info_list
    }
    
    # Guardar reporte
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Reporte guardado en: {report_path}")
    
    return report

def main():
    """Función principal del test simple"""
    logger.info("🎬 INICIANDO TEST SIMPLE DE VIDEOS")
    logger.info("=" * 50)
    
    # Verificar OpenCV
    if not test_opencv_installation():
        logger.error("❌ OpenCV no está funcionando correctamente")
        return False
    
    # Crear directorios
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    FRAMES_PATH.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existen los videos
    logger.info("\n📂 Verificando archivos de video...")
    video_info_list = []
    
    for camera_id, video_path in VIDEO_PATHS.items():
        info = analyze_video(video_path, camera_id)
        video_info_list.append(info)
    
    # Generar reporte
    logger.info("\n📊 Generando reporte...")
    report = generate_test_report(video_info_list)
    
    # Mostrar resumen
    logger.info("\n" + "=" * 50)
    logger.info("📊 RESUMEN DEL TEST")
    logger.info("=" * 50)
    
    summary = report['summary']
    logger.info(f"📹 Videos analizados: {summary['total_videos']}")
    logger.info(f"✅ Videos legibles: {summary['readable_videos']}")
    logger.info(f"⏱️ Duración total: {summary['total_duration_seconds']}s")
    logger.info(f"🎞️ Frames totales: {summary['total_frames']}")
    logger.info(f"💾 Tamaño total: {summary['total_size_mb']} MB")
    logger.info(f"🎯 FPS promedio: {summary['average_fps']}")
    
    # Mostrar detalles por cámara
    logger.info("\n📹 Detalles por cámara:")
    for info in video_info_list:
        status = "✅" if info['readable'] else "❌"
        logger.info(f"   {status} {info['camera_id']}: {info['width']}x{info['height']}, "
                   f"{info['fps']:.1f} FPS, {info['duration_seconds']}s, "
                   f"{info['file_size_mb']} MB")
    
    # Verificar si se extrajeron frames
    total_sample_frames = sum(info['sample_frames_extracted'] for info in video_info_list)
    logger.info(f"\n🖼️ Frames de muestra extraídos: {total_sample_frames}")
    
    if total_sample_frames > 0:
        logger.info(f"📁 Frames guardados en: {FRAMES_PATH}")
        
        # Listar frames extraídos
        for camera_id in VIDEO_PATHS.keys():
            camera_frames_dir = FRAMES_PATH / camera_id
            if camera_frames_dir.exists():
                frames = list(camera_frames_dir.glob("*.jpg"))
                if frames:
                    logger.info(f"   {camera_id}: {len(frames)} frames")
    
    logger.info("\n🎉 TEST SIMPLE COMPLETADO")
    
    # Determinar éxito
    success = summary['readable_videos'] > 0
    if success:
        logger.info("✅ Al menos un video se pudo procesar correctamente")
    else:
        logger.error("❌ Ningún video se pudo procesar")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

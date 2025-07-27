"""
API principal Flask para el servidor de procesamiento de video
Sistema de análisis de marcha para detección de gonartrosis
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar configuraciones
from config import server_config, data_config
from backend.processing import processing_coordinator
from backend.processing.pipeline import video_pipeline, initialize_pipeline
from backend.reconstruction.triangulation import Triangulator
from backend.reconstruction.calibration import calibration_system

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)

# Configuración Flask
app.config['MAX_CONTENT_LENGTH'] = server_config.max_content_length
app.config['UPLOAD_FOLDER'] = server_config.upload_folder

# Inicializar directorios
data_config.ensure_directories()

# Variable global para sesión actual
current_session = {
    'patient_id': None,
    'session_id': None,
    'is_active': False,
    'cameras_count': 0
}

def _check_and_trigger_3d_reconstruction(patient_id: str, session_id: str, chunk_number: int):
    """
    Verificar si tenemos todos los keypoints 2D necesarios para reconstrucción 3D
    y activar triangulación si están disponibles
    """
    try:
        # Verificar si tenemos keypoints 2D procesados de todas las cámaras para este chunk
        session_base = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}"
        
        # Verificar resultados de procesamiento 2D
        keypoints_available = True
        camera_keypoints = {}
        
        # Calcular global_frame (mismo cálculo que en el coordinador)
        global_frame_base = chunk_number * 1000  # Frames base para este chunk
        
        for camera_id in range(current_session['cameras_count']):
            camera_dir = session_base / f"camera{camera_id}"
            
            if camera_dir.exists():
                # Buscar archivos de keypoints .npy para este chunk
                keypoint_files = list(camera_dir.glob(f"{global_frame_base}*_*.npy"))
                # Filtrar solo archivos de keypoints (no confidence)
                keypoint_files = [f for f in keypoint_files if "_confidence" not in f.name]
                
                if keypoint_files:
                    camera_keypoints[camera_id] = keypoint_files
                    logger.debug(f"Cámara {camera_id}: {len(keypoint_files)} archivos de keypoints encontrados")
                else:
                    keypoints_available = False
                    logger.debug(f"Cámara {camera_id}: No se encontraron keypoints para chunk {chunk_number}")
                    break
            else:
                keypoints_available = False
                logger.debug(f"Directorio cámara {camera_id} no existe")
                break
        
        if keypoints_available and len(camera_keypoints) == current_session['cameras_count']:
            logger.info(f"🔺 Iniciando reconstrucción 3D para chunk {chunk_number}")
            
            # Llamar al triangulador
            try:
                triangulator = Triangulator()
                
                # Configurar triangulador (por ahora simple, sin cámaras calibradas)
                result_3d = _triangulate_chunk_simple(
                    triangulator, camera_keypoints, patient_id, session_id, chunk_number
                )
                
                if result_3d:
                    logger.info(f"Reconstrucción 3D completada para chunk {chunk_number}")
                else:
                    logger.warning(f"⚠️  Reconstrucción 3D falló para chunk {chunk_number}")
                    
            except Exception as triangulation_error:
                logger.error(f"Error en triangulación: {triangulation_error}")
        else:
            logger.debug(f"Reconstrucción 3D no disponible aún para chunk {chunk_number}: "
                        f"{len(camera_keypoints)}/{current_session['cameras_count']} cámaras")
            
    except Exception as e:
        logger.error(f"Error verificando reconstrucción 3D: {e}")


def _triangulate_chunk_simple(triangulator: Triangulator, camera_keypoints: Dict[int, list], 
                            patient_id: str, session_id: str, chunk_number: int) -> bool:
    """
    Realizar triangulación simple para un chunk
    """
    try:
        from config import data_config
        
        # Procesar cada frame del chunk
        processed_frames = 0
        
        # Agrupar archivos por global_frame
        frames_data = {}
        
        for camera_id, keypoint_files in camera_keypoints.items():
            for file_path in keypoint_files:
                # Extraer global_frame del nombre del archivo
                filename = file_path.stem  # Nombre sin extensión
                parts = filename.split('_')
                if len(parts) >= 2:
                    global_frame = int(parts[0])
                    detector_name = '_'.join(parts[1:])
                    
                    if global_frame not in frames_data:
                        frames_data[global_frame] = {}
                    
                    if camera_id not in frames_data[global_frame]:
                        frames_data[global_frame][camera_id] = {}
                    
                    # Cargar keypoints
                    keypoints = np.load(file_path)
                    frames_data[global_frame][camera_id][detector_name] = keypoints
        
        # Procesar cada frame
        results_3d = []
        
        for global_frame, cameras_data in frames_data.items():
            if len(cameras_data) == current_session['cameras_count']:  # Todas las cámaras disponibles
                
                # Por simplicidad, usar el primer detector disponible
                detector_names = set()
                for cam_data in cameras_data.values():
                    detector_names.update(cam_data.keys())
                
                if detector_names:
                    primary_detector = list(detector_names)[0]
                    
                    # Extraer keypoints de todas las cámaras para este detector
                    multi_camera_keypoints = {}
                    for camera_id, cam_data in cameras_data.items():
                        if primary_detector in cam_data:
                            multi_camera_keypoints[camera_id] = cam_data[primary_detector]
                    
                    if len(multi_camera_keypoints) >= 2:  # Mínimo 2 cámaras para triangulación
                        # Triangulación simple (placeholder)
                        # TODO: Implementar triangulación real cuando tengamos calibración
                        
                        # Guardar resultado 3D en .json
                        output_dir = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}" / "3D"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        result_3d_data = {
                            'patient_id': patient_id,
                            'session_id': session_id,
                            'chunk_number': chunk_number,
                            'global_frame': global_frame,
                            'timestamp': global_frame / 30.0,  # Asumiendo 30 FPS para timestamp aproximado
                            'detector_used': primary_detector,
                            'cameras_used': list(multi_camera_keypoints.keys()),
                            'num_cameras': len(multi_camera_keypoints),
                            'triangulation_method': 'simple_placeholder',
                            'points_3d_shape': [len(list(multi_camera_keypoints.values())[0]), 3],
                            'status': 'placeholder_generated'
                        }
                        
                        output_file = output_dir / f"frame_{global_frame}_3d.json"
                        with open(output_file, 'w') as f:
                            json.dump(result_3d_data, f, indent=2)
                        
                        processed_frames += 1
                        results_3d.append(output_file)
        
        logger.info(f"💎 Triangulación completada: {processed_frames} frames procesados para chunk {chunk_number}")
        return processed_frames > 0
        
    except Exception as e:
        logger.error(f"Error en triangulación simple: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud del servidor"""
    return jsonify({
        'status': 'healthy',
        'service': 'gonarthrosis-analysis-server',
        'version': '1.0.0'
    })

@app.route('/api/session/start', methods=['POST'])
def start_session():
    """
    Iniciar nueva sesión de procesamiento
    
    Body:
    {
        "patient_id": "string",
        "session_id": "string", 
        "cameras_count": int
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        patient_id = data.get('patient_id')
        session_id = data.get('session_id') 
        cameras_count = data.get('cameras_count', 3)
        
        if not patient_id or not session_id:
            return jsonify({'error': 'patient_id and session_id are required'}), 400
        
        # Verificar si ya hay una sesión activa
        if current_session['is_active']:
            return jsonify({
                'error': 'Session already active',
                'current_session': {
                    'patient_id': current_session['patient_id'],
                    'session_id': current_session['session_id']
                }
            }), 409
        
        # Crear directorios para la nueva sesión
        session_dirs = []
        for camera_id in range(cameras_count):
            base_dir = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
            base_dir.mkdir(parents=True, exist_ok=True)
            session_dirs.append(str(base_dir))
        
        # Activar sesión
        current_session.update({
            'patient_id': patient_id,
            'session_id': session_id,
            'is_active': True,
            'cameras_count': cameras_count
        })
        
        logger.info(f"Sesión iniciada - Paciente: {patient_id}, Sesión: {session_id}, Cámaras: {cameras_count}")
        
        return jsonify({
            'status': 'session_started',
            'patient_id': patient_id,
            'session_id': session_id,
            'cameras_count': cameras_count,
            'directories_created': session_dirs
        })
        
    except Exception as e:
        logger.error(f"Error iniciando sesión: {str(e)}")
        return jsonify({'error': f'Failed to start session: {str(e)}'}), 500

@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    """Obtener estado de la sesión actual"""
    return jsonify({
        'session_active': current_session['is_active'],
        'current_session': current_session if current_session['is_active'] else None
    })

@app.route('/api/session/cancel', methods=['POST'])
def cancel_session():
    """
    Cancelar sesión actual y limpiar datos
    """
    try:
        if not current_session['is_active']:
            return jsonify({'error': 'No active session to cancel'}), 400
        
        patient_id = current_session['patient_id']
        session_id = current_session['session_id']
        
        # Limpiar directorios de la sesión cancelada
        session_path = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}"
        
        if session_path.exists():
            import shutil
            shutil.rmtree(session_path)
            logger.info(f"Directorio de sesión eliminado: {session_path}")
        
        # También limpiar datos procesados si existen
        processed_paths = [
            data_config.photos_dir / f"patient{patient_id}" / f"session{session_id}",
            data_config.keypoints_2d_dir / f"patient{patient_id}" / f"session{session_id}",
            data_config.keypoints_3d_dir / f"patient{patient_id}" / f"session{session_id}"
        ]
        
        for path in processed_paths:
            if path.exists():
                import shutil
                shutil.rmtree(path)
                logger.info(f"Directorio procesado eliminado: {path}")
        
        # Reiniciar sesión
        logger.info(f"Sesión cancelada - Paciente: {patient_id}, Sesión: {session_id}")
        
        old_session = current_session.copy()
        current_session.update({
            'patient_id': None,
            'session_id': None,
            'is_active': False,
            'cameras_count': 0
        })
        
        return jsonify({
            'status': 'session_cancelled',
            'cancelled_session': {
                'patient_id': old_session['patient_id'],
                'session_id': old_session['session_id']
            }
        })
        
    except Exception as e:
        logger.error(f"Error cancelando sesión: {str(e)}")
        return jsonify({'error': f'Failed to cancel session: {str(e)}'}), 500

@app.route('/api/chunks/receive', methods=['POST'])
def receive_chunk():
    """
    Recibir chunk de video para procesamiento
    
    Form data:
    - file: archivo de video (.mp4)
    - camera_id: ID de la cámara (0, 1, 2...)
    - chunk_number: número del chunk
    """
    try:
        # Verificar sesión activa
        if not current_session['is_active']:
            return jsonify({'error': 'No active session'}), 400
        
        # Verificar archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Obtener metadatos
        camera_id = request.form.get('camera_id')
        chunk_number = request.form.get('chunk_number')
        
        if camera_id is None or chunk_number is None:
            return jsonify({'error': 'camera_id and chunk_number are required'}), 400
        
        try:
            camera_id = int(camera_id)
            chunk_number = int(chunk_number)
        except ValueError:
            return jsonify({'error': 'camera_id and chunk_number must be integers'}), 400
        
        # Verificar que la cámara esté en rango
        if camera_id >= current_session['cameras_count']:
            return jsonify({'error': f'Invalid camera_id. Max: {current_session["cameras_count"]-1}'}), 400
        
        # Guardar archivo
        patient_id = current_session['patient_id']
        session_id = current_session['session_id']
        
        save_dir = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{chunk_number}.mp4"
        file_path = save_dir / filename
        
        file.save(str(file_path))
        
        logger.info(f"Chunk recibido - Cámara: {camera_id}, Chunk: {chunk_number}, Tamaño: {file_path.stat().st_size} bytes")
        
        # Verificar si tenemos todos los chunks de todas las cámaras para este número
        try:
            session_base = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}"
            available_chunks = {}
            
            # Verificar chunks disponibles para este número de chunk
            for cam_id in range(current_session['cameras_count']):
                chunk_path = session_base / f"camera{cam_id}" / f"{chunk_number}.mp4"
                if chunk_path.exists():
                    available_chunks[cam_id] = chunk_path
            
            # Si tenemos todos los chunks, procesar inmediatamente
            if len(available_chunks) == current_session['cameras_count']:
                logger.info(f"Todos los chunks {chunk_number} disponibles, procesando inmediatamente...")
                
                # Inicializar pipeline si no está inicializado
                if not video_pipeline.is_initialized:
                    if not initialize_pipeline():
                        logger.error("Error inicializando video_pipeline")
                        raise Exception("Video pipeline initialization failed")
                
                # Procesar chunk sincronizado usando el pipeline
                result = video_pipeline.process_chunk_synchronized(
                    patient_id=patient_id,
                    session_id=session_id,
                    chunk_number=chunk_number,
                    video_paths=available_chunks
                )
                
                if result.success:
                    logger.info(f"Chunk {chunk_number} procesado: {result.total_frames} frames, "
                               f"tiempo: {result.processing_time:.2f}s")
                    
                    # Verificar si podemos hacer reconstrucción 3D
                    _check_and_trigger_3d_reconstruction(patient_id, session_id, chunk_number)
                    
                else:
                    logger.error(f"Error procesando chunk {chunk_number}: {result.errors}")
                    
        except Exception as processing_error:
            logger.error(f"Error en procesamiento inmediato: {processing_error}")
        
        return jsonify({
            'status': 'chunk_received',
            'camera_id': camera_id,
            'chunk_number': chunk_number,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size
        })
        
    except Exception as e:
        logger.error(f"Error recibiendo chunk: {str(e)}")
        return jsonify({'error': f'Failed to receive chunk: {str(e)}'}), 500

@app.route('/api/cameras/recalibrate', methods=['POST'])
def recalibrate_cameras():
    """
    Recalcular parámetros extrínsecos de las cámaras
    Se ejecuta cuando se mueven las cámaras
    """
    try:
        if not current_session['is_active']:
            return jsonify({'error': 'No active session'}), 400
        
        patient_id = current_session['patient_id']
        session_id = current_session['session_id']
        
        logger.info(f"Iniciando recalibración de cámaras para sesión {patient_id}/{session_id}...")
        
        # Usar sistema de calibración para auto-calibrar usando keypoints de la sesión
        calibration_result = calibration_system.auto_calibrate_extrinsics_from_session(
            patient_id=patient_id,
            session_id=session_id
        )
        
        if 'error' in calibration_result:
            logger.error(f"Error en auto-calibración: {calibration_result['error']}")
            return jsonify({
                'success': False,
                'error': calibration_result['error']
            }), 500
        
        if calibration_result.get('success', False):
            logger.info(f"Recalibración exitosa: {calibration_result['calibrated_count']} cámaras calibradas")
            
            # Guardar calibración actualizada
            try:
                calibration_file = data_config.base_data_dir / f"patient{patient_id}" / f"session{session_id}" / "calibration.npz"
                calibration_file.parent.mkdir(parents=True, exist_ok=True)
                calibration_system.save_calibration(str(calibration_file))
                logger.info(f"Calibración guardada en: {calibration_file}")
            except Exception as save_error:
                logger.warning(f"Error guardando calibración: {save_error}")
            
            return jsonify({
                'success': True,
                'status': 'recalibration_completed',
                'message': 'Camera extrinsic parameters recalibrated successfully',
                'calibration_result': calibration_result
            })
        else:
            logger.warning("Recalibración falló")
            return jsonify({
                'success': False,
                'status': 'recalibration_failed',
                'message': 'Camera recalibration failed',
                'calibration_result': calibration_result
            }), 400
        
    except Exception as e:
        logger.error(f"Error en recalibración: {str(e)}")
        return jsonify({'error': f'Failed to recalibrate cameras: {str(e)}'}), 500

@app.route('/api/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """
    Obtener estado completo del pipeline de procesamiento
    """
    try:
        status = video_pipeline.get_processing_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error obteniendo estado del pipeline: {str(e)}")
        return jsonify({'error': f'Failed to get pipeline status: {str(e)}'}), 500

@app.route('/api/session/<patient_id>/<session_id>/analysis', methods=['GET'])
def get_session_analysis(patient_id: str, session_id: str):
    """
    Obtener análisis completo de una sesión procesada
    
    Args:
        patient_id: ID del paciente
        session_id: ID de la sesión
    """
    try:
        analysis = video_pipeline.get_session_analysis(patient_id, session_id)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error obteniendo análisis de sesión: {str(e)}")
        return jsonify({'error': f'Failed to get session analysis: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Manejar archivos demasiado grandes"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    """Manejar rutas no encontradas"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Manejar errores internos"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info(f"Iniciando servidor de análisis de gonartrosis...")
    logger.info(f"Puerto: {server_config.port}")
    logger.info(f"Directorio de datos: {data_config.base_data_dir}")
    
    app.run(
        host=server_config.host,
        port=server_config.port, 
        debug=server_config.debug
    )

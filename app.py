"""
API principal Flask para el servidor de procesamiento de video
Sistema de análisis de marcha para detección de gonartrosis
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar configuraciones
from config import server_config, data_config

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

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud del servidor"""
    return jsonify({
        'status': 'healthy',
        'service': 'gonitrosis-analysis-server',
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
            
            # Crear subdirectorios para datos 2D
            points_dir = base_dir / "2D" / "points"
            confidence_dir = base_dir / "2D" / "confidence"
            
            # Crear directorios para cada modelo
            from config import processing_config
            all_models = processing_config.coco_models + processing_config.extended_models
            
            for model_name in all_models:
                (points_dir / model_name).mkdir(parents=True, exist_ok=True)
                (confidence_dir / model_name).mkdir(parents=True, exist_ok=True)
            
            session_dirs.append(str(base_dir))
        
        # Activar sesión
        current_session.update({
            'patient_id': patient_id,
            'session_id': session_id,
            'is_active': True,
            'cameras_count': cameras_count
        })
        
        logger.info(f"📋 Sesión iniciada - Paciente: {patient_id}, Sesión: {session_id}, Cámaras: {cameras_count}")
        
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
            logger.info(f"🗑️ Directorio de sesión eliminado: {session_path}")
        
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
                logger.info(f"🗑️ Directorio procesado eliminado: {path}")
        
        # Reiniciar sesión
        logger.info(f"❌ Sesión cancelada - Paciente: {patient_id}, Sesión: {session_id}")
        
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
        
        logger.info(f"📹 Chunk recibido - Cámara: {camera_id}, Chunk: {chunk_number}, Tamaño: {file_path.stat().st_size} bytes")
        
        # TODO: Iniciar procesamiento asíncrono del chunk
        # process_video_chunk.delay(str(file_path), patient_id, session_id, camera_id, chunk_number)
        
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
        
        # TODO: Implementar recalibración de parámetros extrínsecos
        # usando keypoints 2D de múltiples frames
        
        logger.info("🔧 Iniciando recalibración de cámaras...")
        
        return jsonify({
            'status': 'recalibration_started',
            'message': 'Camera extrinsic parameters recalibration initiated'
        })
        
    except Exception as e:
        logger.error(f"Error en recalibración: {str(e)}")
        return jsonify({'error': f'Failed to recalibrate cameras: {str(e)}'}), 500

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
    logger.info(f"🚀 Iniciando servidor de análisis de gonartrosis...")
    logger.info(f"📡 Puerto: {server_config.port}")
    logger.info(f"📁 Directorio de datos: {data_config.base_data_dir}")
    
    app.run(
        host=server_config.host,
        port=server_config.port, 
        debug=server_config.debug
    )

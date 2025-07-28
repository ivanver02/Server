"""
Procesador de ensemble learning para fusionar keypoints 2D de múltiples modelos
Implementa estrategias de fusión ponderada por confianza para máxima precisión
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EnsembleResult:
    """Resultado del ensemble learning"""
    keypoints_2d: np.ndarray  # (N, 2) coordenadas finales
    confidence_scores: np.ndarray  # (N,) confianza final
    model_contributions: Dict[str, float]  # Contribución de cada modelo
    processing_info: Dict[str, Any]  # Info adicional

class EnsembleProcessor:
    """
    Procesador de ensemble learning para keypoints 2D
    Combina resultados de múltiples modelos MMPose para máxima precisión
    """
    
    def __init__(self):
        self.coco_models = []  # Modelos COCO (17 keypoints)
        self.extended_models = []  # Modelos con keypoints adicionales
        self.ensemble_weights = {}
        self.confidence_threshold = 0.3
        
        # Cargar configuración
        self._load_config()
    
    def _load_config(self):
        """Cargar configuración de ensemble"""
        try:
            from config import processing_config
            
            self.coco_models = processing_config.coco_models
            self.extended_models = processing_config.extended_models
            self.ensemble_weights = processing_config.ensemble_weights
            self.confidence_threshold = processing_config.confidence_threshold
            
            logger.info(f"Ensemble configurado - COCO: {self.coco_models}, Extendidos: {self.extended_models}")
            
        except Exception as e:
            logger.error(f"Error cargando configuración ensemble: {e}")
            # Configuración por defecto usando nombres modernos
            self.coco_models = ['hrnet_w48', 'vitpose']
            self.extended_models = ['resnet50_rle', 'wholebody']
            self.ensemble_weights = {
                'hrnet_w48': 0.6,
                'vitpose': 0.4,
                'resnet50_rle': 1.0,
                'wholebody': 1.0
            }
    
    def load_keypoints_for_frame(self, patient_id: str, session_id: str, 
                                camera_id: int, global_frame: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Cargar keypoints de todos los modelos para un frame específico
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            global_frame: Número global del frame
            
        Returns:
            Dict {model_name: (keypoints, confidence)}
        """
        keypoints_data = {}
        
        try:
            from config import data_config
            
            base_dir = (data_config.unprocessed_dir / 
                       f"patient{patient_id}" / 
                       f"session{session_id}" / 
                       f"camera{camera_id}" / 
                       "2D")
            
            all_models = self.coco_models + self.extended_models
            
            for model_name in all_models:
                points_file = base_dir / "points" / model_name / f"{global_frame}.npy"
                confidence_file = base_dir / "confidence" / model_name / f"{global_frame}.npy"
                
                if points_file.exists() and confidence_file.exists():
                    try:
                        keypoints = np.load(points_file)
                        confidence = np.load(confidence_file)
                        
                        # Validar dimensiones
                        if len(keypoints.shape) == 2 and keypoints.shape[1] == 2:
                            if len(confidence) == len(keypoints):
                                keypoints_data[model_name] = (keypoints, confidence)
                            else:
                                logger.warning(f"Dimensiones inconsistentes para {model_name} frame {global_frame}")
                        else:
                            logger.warning(f"Formato inválido de keypoints para {model_name} frame {global_frame}")
                            
                    except Exception as e:
                        logger.error(f"Error cargando {model_name} frame {global_frame}: {e}")
                else:
                    logger.debug(f"Archivos no encontrados para {model_name} frame {global_frame}")
            
            logger.debug(f"Cargados keypoints para frame {global_frame}: {list(keypoints_data.keys())}")
            return keypoints_data
            
        except Exception as e:
            logger.error(f"Error cargando keypoints para frame {global_frame}: {e}")
            return {}
    
    def fuse_coco_keypoints(self, coco_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fusionar keypoints COCO (17 puntos) usando promedio ponderado por confianza
        
        Args:
            coco_data: Dict {model_name: (keypoints, confidence)} para modelos COCO
            
        Returns:
            Tuple (fused_keypoints, fused_confidence)
        """
        if not coco_data:
            return np.zeros((17, 2)), np.zeros(17)
        
        try:
            # Preparar datos para fusión
            all_keypoints = []
            all_confidences = []
            model_weights = []
            
            for model_name, (keypoints, confidence) in coco_data.items():
                if len(keypoints) >= 17:  # Verificar que tiene al menos los 17 puntos COCO
                    coco_keypoints = keypoints[:17]  # Tomar solo los primeros 17
                    coco_confidence = confidence[:17]
                    
                    all_keypoints.append(coco_keypoints)
                    all_confidences.append(coco_confidence)
                    
                    # Peso del modelo desde configuración
                    weight = self.ensemble_weights.get(model_name, 1.0)
                    model_weights.append(weight)
            
            if not all_keypoints:
                return np.zeros((17, 2)), np.zeros(17)
            
            # Convertir a arrays numpy
            all_keypoints = np.array(all_keypoints)  # (num_models, 17, 2)
            all_confidences = np.array(all_confidences)  # (num_models, 17)
            model_weights = np.array(model_weights)  # (num_models,)
            
            # Calcular pesos finales: peso_modelo * confianza
            # Expandir model_weights para broadcasting
            model_weights_expanded = model_weights[:, np.newaxis]  # (num_models, 1)
            final_weights = model_weights_expanded * all_confidences  # (num_models, 17)
            
            # Normalizar pesos por keypoint
            weight_sums = np.sum(final_weights, axis=0)  # (17,)
            weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)  # Evitar división por 0
            
            normalized_weights = final_weights / weight_sums[np.newaxis, :]  # (num_models, 17)
            
            # Calcular keypoints fusionados
            weighted_keypoints = normalized_weights[:, :, np.newaxis] * all_keypoints  # (num_models, 17, 2)
            fused_keypoints = np.sum(weighted_keypoints, axis=0)  # (17, 2)
            
            # Calcular confianza fusionada (promedio ponderado)
            fused_confidence = np.sum(final_weights, axis=0) / len(all_keypoints)  # (17,)
            
            logger.debug(f"COCO fusionado: {len(coco_data)} modelos, confianza media: {np.mean(fused_confidence):.3f}")
            
            return fused_keypoints, fused_confidence
            
        except Exception as e:
            logger.error(f"Error fusionando keypoints COCO: {e}")
            return np.zeros((17, 2)), np.zeros(17)
    
    def extract_additional_keypoints(self, extended_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extraer keypoints adicionales de modelos extendidos (no COCO)
        
        Args:
            extended_data: Dict {model_name: (keypoints, confidence)} para modelos extendidos
            
        Returns:
            Tuple (additional_keypoints, additional_confidence, keypoint_names)
        """
        additional_keypoints = []
        additional_confidence = []
        keypoint_names = []
        
        try:
            from config import get_gonartrosis_indices
            
            # Procesar cada modelo extendido
            for model_name, (keypoints, confidence) in extended_data.items():
                # Obtener índices de keypoints relevantes para gonartrosis
                relevant_indices = get_gonartrosis_indices(model_name)
                
                if 'wholebody' in model_name.lower():
                    # Para wholebody, extraer keypoints de pies (índices 127-138)
                    if len(keypoints) >= 133:  # WholeBody tiene 133 keypoints
                        # Keypoints de pies (índices 127-132 para pies)
                        foot_indices = list(range(127, 133))  # 6 keypoints de pies
                        
                        for idx in foot_indices:
                            if idx < len(keypoints):
                                additional_keypoints.append(keypoints[idx])
                                additional_confidence.append(confidence[idx])
                                
                                # Nombres descriptivos
                                foot_names = [
                                    'left_big_toe', 'left_small_toe', 'left_heel',
                                    'right_big_toe', 'right_small_toe', 'right_heel'
                                ]
                                if len(keypoint_names) < len(foot_names):
                                    keypoint_names.append(foot_names[len(keypoint_names)])
                
                elif 'resnet50' in model_name.lower():
                    # Para ResNet50, agregar keypoints únicos si los hay
                    # Por ahora, ResNet50 también es COCO, pero podría tener procesamiento especial
                    pass
            
            if additional_keypoints:
                additional_keypoints = np.array(additional_keypoints)
                additional_confidence = np.array(additional_confidence)
            else:
                additional_keypoints = np.zeros((0, 2))
                additional_confidence = np.array([])
                keypoint_names = []
            
            logger.debug(f"Keypoints adicionales extraídos: {len(additional_keypoints)}")
            
            return additional_keypoints, additional_confidence, keypoint_names
            
        except Exception as e:
            logger.error(f"Error extrayendo keypoints adicionales: {e}")
            return np.zeros((0, 2)), np.array([]), []
    
    def process_frame_ensemble(self, patient_id: str, session_id: str, 
                              camera_id: int, global_frame: int) -> Optional[EnsembleResult]:
        """
        Procesar ensemble completo para un frame
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            global_frame: Número global del frame
            
        Returns:
            Resultado del ensemble o None si error
        """
        try:
            # 1. Cargar keypoints de todos los modelos
            all_keypoints_data = self.load_keypoints_for_frame(
                patient_id, session_id, camera_id, global_frame
            )
            
            if not all_keypoints_data:
                logger.warning(f"No hay datos de keypoints para frame {global_frame}")
                return None
            
            # 2. Separar modelos COCO y extendidos
            coco_data = {k: v for k, v in all_keypoints_data.items() if k in self.coco_models}
            extended_data = {k: v for k, v in all_keypoints_data.items() if k in self.extended_models}
            
            # 3. Fusionar keypoints COCO
            coco_keypoints, coco_confidence = self.fuse_coco_keypoints(coco_data)
            
            # 4. Extraer keypoints adicionales
            additional_keypoints, additional_confidence, additional_names = self.extract_additional_keypoints(extended_data)
            
            # 5. Combinar todos los keypoints
            if len(additional_keypoints) > 0:
                final_keypoints = np.vstack([coco_keypoints, additional_keypoints])
                final_confidence = np.concatenate([coco_confidence, additional_confidence])
            else:
                final_keypoints = coco_keypoints
                final_confidence = coco_confidence
            
            # 6. Filtrar por confianza mínima
            valid_mask = final_confidence >= self.confidence_threshold
            final_keypoints = final_keypoints[valid_mask]
            final_confidence = final_confidence[valid_mask]
            
            # 7. Calcular contribuciones de modelos
            model_contributions = {}
            for model_name in all_keypoints_data.keys():
                weight = self.ensemble_weights.get(model_name, 1.0)
                model_contributions[model_name] = weight
            
            # 8. Crear resultado
            result = EnsembleResult(
                keypoints_2d=final_keypoints,
                confidence_scores=final_confidence,
                model_contributions=model_contributions,
                processing_info={
                    'frame_id': global_frame,
                    'camera_id': camera_id,
                    'total_keypoints': len(final_keypoints),
                    'coco_keypoints': len(coco_keypoints),
                    'additional_keypoints': len(additional_keypoints),
                    'models_used': list(all_keypoints_data.keys()),
                    'mean_confidence': float(np.mean(final_confidence)) if len(final_confidence) > 0 else 0.0
                }
            )
            
            logger.debug(f"Ensemble completado para frame {global_frame}: "
                        f"{len(final_keypoints)} keypoints, confianza: {result.processing_info['mean_confidence']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en ensemble para frame {global_frame}: {e}")
            return None
    
    def save_ensemble_result(self, result: EnsembleResult, patient_id: str, 
                           session_id: str, camera_id: int, global_frame: int) -> bool:
        """
        Guardar resultado del ensemble en disco
        
        Args:
            result: Resultado del ensemble
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            global_frame: Número global del frame
            
        Returns:
            True si se guardó correctamente
        """
        try:
            from config import data_config
            
            # Directorio destino
            save_dir = (data_config.keypoints_2d_dir / 
                       f"patient{patient_id}" / 
                       f"session{session_id}" / 
                       f"camera{camera_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar keypoints y confianza
            keypoints_file = save_dir / f"{global_frame}.npy"
            confidence_file = save_dir / f"{global_frame}_confidence.npy"
            
            np.save(keypoints_file, result.keypoints_2d)
            np.save(confidence_file, result.confidence_scores)
            
            logger.debug(f"Ensemble guardado: {keypoints_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando ensemble: {e}")
            return False
    
    def process_complete_session(self, patient_id: str, session_id: str) -> Dict[str, Any]:
        """
        Procesar ensemble para toda una sesión
        
        Args:
            patient_id: ID del paciente
            session_id: ID de la sesión
            
        Returns:
            Resumen del procesamiento
        """
        logger.info(f"Iniciando ensemble para sesión {patient_id}/{session_id}")
        
        try:
            from config import data_config
            
            session_dir = (data_config.unprocessed_dir / 
                          f"patient{patient_id}" / 
                          f"session{session_id}")
            
            if not session_dir.exists():
                return {'error': f'Sesión no encontrada: {session_dir}'}
            
            processed_frames = 0
            total_cameras = 0
            errors = []
            
            # Procesar cada cámara
            for camera_dir in session_dir.iterdir():
                if camera_dir.is_dir() and camera_dir.name.startswith('camera'):
                    camera_id = int(camera_dir.name.replace('camera', ''))
                    total_cameras += 1
                    
                    # Encontrar todos los frames procesados
                    points_dir = camera_dir / "2D" / "points"
                    if points_dir.exists():
                        # Obtener frames disponibles (buscar en el primer modelo)
                        first_model_dir = None
                        for model_dir in points_dir.iterdir():
                            if model_dir.is_dir():
                                first_model_dir = model_dir
                                break
                        
                        if first_model_dir:
                            frame_files = list(first_model_dir.glob("*.npy"))
                            frame_numbers = sorted([int(f.stem) for f in frame_files])
                            
                            # Procesar cada frame
                            for frame_num in frame_numbers:
                                result = self.process_frame_ensemble(
                                    patient_id, session_id, camera_id, frame_num
                                )
                                
                                if result:
                                    if self.save_ensemble_result(
                                        result, patient_id, session_id, camera_id, frame_num
                                    ):
                                        processed_frames += 1
                                    else:
                                        errors.append(f"Error guardando frame {frame_num} cámara {camera_id}")
                                else:
                                    errors.append(f"Error procesando frame {frame_num} cámara {camera_id}")
            
            summary = {
                'patient_id': patient_id,
                'session_id': session_id,
                'processed_frames': processed_frames,
                'cameras_processed': total_cameras,
                'errors': errors,
                'success': len(errors) == 0 and processed_frames > 0
            }
            
            logger.info(f"✅ Ensemble completado: {processed_frames} frames, {total_cameras} cámaras")
            
            return summary
            
        except Exception as e:
            error_msg = f"Error en ensemble de sesión: {e}"
            logger.error(error_msg)
            return {'error': error_msg}

# Instancia global del procesador
ensemble_processor = EnsembleProcessor()

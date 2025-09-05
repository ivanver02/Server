import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class Models2DAnalysis:
    """Análisis comparativo de medidas corporales 2D entre diferentes modelos de pose estimation"""
    
    # Constantes
    CONFIDENCE_THRESHOLD = 0.5
    FOREARM_REFERENCE_CM = 30.0  # 30 cm como referencia del antebrazo
    NUM_KEYPOINTS = 23  # Solo analizamos los primeros 23 keypoints de los 133
    
    def __init__(self, patient_id: str, session_id: str, frame_id: int, chunk_id: int):
        self.patient_id = patient_id
        self.session_id = session_id
        self.frame_id = frame_id
        self.chunk_id = chunk_id
        
        # Ruta base de datos - usando ruta absoluta
        self.data_root = Path(r"c:\Users\usuario\Desktop\Universidad\TERCERO\Laboratorio Julio\Definitivo\Server\data\unprocessed")
        self.keypoints_dir = self.data_root / patient_id / session_id / "keypoints2D"
        
        # Verificar que la carpeta existe
        if not self.keypoints_dir.exists():
            raise ValueError(f"No se encuentra la carpeta de keypoints: {self.keypoints_dir}")

    def get_available_models(self) -> List[str]:
        """Obtiene la lista de modelos disponibles en la carpeta de keypoints2D."""
        models = []
        for item in self.keypoints_dir.iterdir():
            if item.is_dir():
                models.append(item.name)
        
        logger.info(f"Modelos disponibles: {models}")
        return sorted(models)

    def load_keypoints_data(self, model_name: str, camera_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Carga los datos de keypoints y confianzas para un modelo y cámara específicos."""
        
        # Construir rutas
        coords_file = self.keypoints_dir / model_name / camera_id / "coordinates" / f"{self.frame_id}_{self.chunk_id}.npy"
        conf_file = self.keypoints_dir / model_name / camera_id / "confidence" / f"{self.frame_id}_{self.chunk_id}.npy"
        
        try:
            # Cargar coordenadas y confianzas
            coords = np.load(coords_file)
            confidences = np.load(conf_file)
            
            # Tomar solo los primeros 23 keypoints
            coords = coords[:self.NUM_KEYPOINTS]
            confidences = confidences[:self.NUM_KEYPOINTS]
            
            logger.info(f"Cargados keypoints para {model_name}/{camera_id}: {coords.shape}")
            return coords, confidences
            
        except FileNotFoundError as e:
            logger.warning(f"Archivo no encontrado para {model_name}/{camera_id}: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Error cargando datos para {model_name}/{camera_id}: {e}")
            return None, None

    def filter_valid_keypoints(self, confidences_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Filtra keypoints que tienen confianza > threshold en todas las cámaras."""
        
        # Obtener todas las confianzas válidas
        conf_arrays = []
        for cam_id in ["camera0", "camera1", "camera2"]:
            if cam_id in confidences_dict and confidences_dict[cam_id] is not None:
                conf_arrays.append(confidences_dict[cam_id] > self.CONFIDENCE_THRESHOLD)
        
        if not conf_arrays:
            return np.zeros(self.NUM_KEYPOINTS, dtype=bool)
        
        # Máscara de puntos válidos en todas las cámaras disponibles
        valid_mask = np.logical_and.reduce(conf_arrays)
        
        return valid_mask

    def calculate_2d_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calcula distancia euclidiana entre dos puntos 2D."""
        return np.linalg.norm(point1 - point2)

    def calculate_scale_factor_from_2d_forearm(self, coords: np.ndarray, valid_mask: np.ndarray, 
                                             real_forearm_length_cm: float = 30.0) -> Tuple[Optional[float], Optional[float]]:
        """Calcula factor de escala para una cámara basado en la longitud del antebrazo en 2D.
        
        Returns:
            Tuple[scale_factor, distance_estimate]: Factor de escala en cm/pixel y distancia estimada en metros
        """
        
        # Índices: 7=Codo_izq, 8=Codo_der, 9=Muñeca_izq, 10=Muñeca_der
        forearm_measurements = []
        
        # Antebrazo izquierdo (codo_izq a muñeca_izq)
        if valid_mask[7] and valid_mask[9]:  # Codo_izq y Muñeca_izq válidos
            left_forearm_2d = self.calculate_2d_distance(coords[7], coords[9])
            forearm_measurements.append(("Izquierdo", left_forearm_2d))
        
        # Antebrazo derecho (codo_der a muñeca_der)
        if valid_mask[8] and valid_mask[10]:  # Codo_der y Muñeca_der válidos
            right_forearm_2d = self.calculate_2d_distance(coords[8], coords[10])
            forearm_measurements.append(("Derecho", right_forearm_2d))
        
        if not forearm_measurements:
            return None, None
        
        # Usar promedio de las medidas disponibles
        avg_forearm_pixels = np.mean([pixels for _, pixels in forearm_measurements])
        scale_factor = real_forearm_length_cm / avg_forearm_pixels  # cm/pixel
        
        # Estimar distancia a la cámara usando parámetros intrínsecos reales
        # Matriz intrínseca de la cámara:
        # fx = 418.3, fy = 417.5, cx = 419.7, cy = 264.2
        # Fórmula: Distancia = (Objeto_real * Focal_length) / Objeto_en_imagen
        fx = 418.3  # Distancia focal en píxeles (eje x)
        fy = 417.5  # Distancia focal en píxeles (eje y)
        focal_length_avg = (fx + fy) / 2  # Promedio para el cálculo de distancia
        
        distance_estimate_cm = (real_forearm_length_cm * focal_length_avg) / avg_forearm_pixels
        distance_estimate_m = distance_estimate_cm / 100  # convertir a metros
        
        return scale_factor, distance_estimate_m

    def analyze_2d_measurements_for_camera(self, coords: np.ndarray, valid_mask: np.ndarray, 
                                         scale_factor: float, model_name: str, camera_id: str):
        """Analiza medidas corporales 2D escaladas para una cámara específica."""
        
        def distance_2d_scaled(p1_idx: int, p2_idx: int) -> float:
            """Calcula distancia 2D escalada entre dos keypoints."""
            if not (valid_mask[p1_idx] and valid_mask[p2_idx]):
                return np.nan
            pixel_distance = self.calculate_2d_distance(coords[p1_idx], coords[p2_idx])
            return pixel_distance * scale_factor  # convertir a cm
        
        # Definir medidas a calcular (mismas que los análisis anteriores)
        measurement_definitions = [
            # Cabeza y cuello
            ("Ancho cara (ojo_izq - ojo_der)", 1, 2, "7-10 cm"),
            # Torso  
            ("Ancho hombros", 5, 6, "35-45 cm"),
            ("Alto torso (hombro_izq - cadera_izq)", 5, 11, "50-70 cm"),
            ("Alto torso (hombro_der - cadera_der)", 6, 12, "50-70 cm"),
            ("Ancho caderas", 11, 12, "25-35 cm"),
            # Brazo izquierdo
            ("Brazo izq (hombro-codo)", 5, 7, "28-36 cm"),
            ("Antebrazo izq (codo-muñeca)", 7, 9, "23-30 cm"),
            ("Brazo completo izq (hombro-muñeca)", 5, 9, "55-70 cm"),
            # Brazo derecho
            ("Brazo der (hombro-codo)", 6, 8, "28-36 cm"),
            ("Antebrazo der (codo-muñeca)", 8, 10, "23-30 cm"),
            ("Brazo completo der (hombro-muñeca)", 6, 10, "55-70 cm"),
            # Pierna izquierda
            ("Muslo izq (cadera-rodilla)", 11, 13, "35-50 cm"),
            ("Pantorrilla izq (rodilla-tobillo)", 13, 15, "35-45 cm"),
            ("Pierna completa izq (cadera-tobillo)", 11, 15, "75-100 cm"),
            # Pierna derecha
            ("Muslo der (cadera-rodilla)", 12, 14, "35-50 cm"),
            ("Pantorrilla der (rodilla-tobillo)", 14, 16, "35-45 cm"),
            ("Pierna completa der (cadera-tobillo)", 12, 16, "75-100 cm"),
            # Medidas adicionales
            ("Estatura aprox (cabeza-tobillo_izq)", 0, 15, "150-190 cm"),
            ("Estatura aprox (cabeza-tobillo_der)", 0, 16, "150-190 cm"),
            ("Envergadura (muñeca_izq - muñeca_der)", 9, 10, "150-180 cm"),
        ]
        
        print(f"\n{'='*100}")
        print(f"ANÁLISIS DE MEDIDAS 2D - MODELO: {model_name.upper()} - CÁMARA: {camera_id.upper()}")
        print(f"Factor de escala: {scale_factor:.4f} cm/pixel")
        print(f"{'='*100}")
        
        print(f"{'Medida':<40} | {'Valor':<12} | {'Rango Normal':<15} | {'Estado'}")
        print("-" * 85)
        
        valid_measurements = 0
        realistic_measurements = 0
        
        for name, p1_idx, p2_idx, normal_range in measurement_definitions:
            value_cm = distance_2d_scaled(p1_idx, p2_idx)
            
            if np.isnan(value_cm):
                status = "❌ N/A"
                value_str = "N/A"
            else:
                value_str = f"{value_cm:.1f} cm"
                
                # Análisis básico de realismo (mismos rangos que análisis anteriores)
                if "cara" in name.lower() or "ojos" in name.lower():
                    realistic = 3 <= value_cm <= 15
                elif "hombros" in name.lower():
                    realistic = 25 <= value_cm <= 55
                elif "torso" in name.lower():
                    realistic = 40 <= value_cm <= 80
                elif "caderas" in name.lower():
                    realistic = 20 <= value_cm <= 45
                elif "brazo" in name.lower() and "completo" not in name.lower():
                    realistic = 20 <= value_cm <= 45
                elif "brazo completo" in name.lower():
                    realistic = 45 <= value_cm <= 80
                elif "muslo" in name.lower():
                    realistic = 30 <= value_cm <= 60
                elif "pantorrilla" in name.lower():
                    realistic = 30 <= value_cm <= 55
                elif "pierna completa" in name.lower():
                    realistic = 65 <= value_cm <= 110
                elif "estatura" in name.lower():
                    realistic = 140 <= value_cm <= 200
                elif "envergadura" in name.lower():
                    realistic = 140 <= value_cm <= 200
                else:
                    realistic = True
                    
                status = "✅ OK" if realistic else "⚠️ Fuera rango"
                if realistic:
                    realistic_measurements += 1
                valid_measurements += 1
            
            print(f"{name:<40} | {value_str:<12} | {normal_range:<15} | {status}")
        
        # Resumen por cámara
        print(f"\n{'='*100}")
        print(f"RESUMEN DE VALIDACIÓN 2D - MODELO: {model_name.upper()} - CÁMARA: {camera_id.upper()}")
        print(f"{'='*100}")
        print(f"Medidas válidas: {valid_measurements}/{len(measurement_definitions)}")
        if valid_measurements > 0:
            percentage = (realistic_measurements / valid_measurements) * 100
            print(f"Medidas realistas: {realistic_measurements}/{valid_measurements} ({percentage:.1f}%)")
            
            if percentage > 80:
                print("🟢 EXCELENTE: Las proporciones 2D son muy realistas")
            elif percentage > 60:
                print("🟡 BUENO: Las proporciones 2D son aceptables") 
            else:
                print("🔴 PROBLEMA: Las proporciones 2D parecen irrealistas")
        else:
            print("❌ No hay medidas válidas para evaluar")
        
        return {
            "valid_measurements": valid_measurements,
            "realistic_measurements": realistic_measurements,
            "realism_percentage": (realistic_measurements / max(valid_measurements, 1)) * 100
        }

    def run_analysis_for_model(self, model_name: str) -> Dict[str, Dict]:
        """Ejecuta el análisis completo para un modelo específico."""
        
        print(f"\n{'#'*120}")
        print(f"ANÁLISIS COMPLETO PARA MODELO: {model_name.upper()}")
        print(f"Paciente: {self.patient_id} | Sesión: {self.session_id} | Frame: {self.frame_id} | Chunk: {self.chunk_id}")
        print(f"{'#'*120}")
        
        # Cargar datos para las 3 cámaras
        cameras_data = {}
        confidences_data = {}
        
        for camera_id in ["camera0", "camera1", "camera2"]:
            coords, confs = self.load_keypoints_data(model_name, camera_id)
            cameras_data[camera_id] = coords
            confidences_data[camera_id] = confs
        
        # Filtrar keypoints válidos
        valid_mask = self.filter_valid_keypoints(confidences_data)
        valid_count = np.sum(valid_mask)
        
        print(f"\n=== FILTRADO DE KEYPOINTS 2D (MODELO: {model_name.upper()}) ===")
        print(f"Umbral de confianza: {self.CONFIDENCE_THRESHOLD}")
        print(f"Puntos válidos en todas las cámaras: {valid_count}/{self.NUM_KEYPOINTS}")
        
        if valid_count == 0:
            print(f"❌ ERROR: No hay puntos válidos para el modelo {model_name}")
            return {}
        
        # Calcular factor de escala para cada cámara y mostrar información detallada
        print(f"\n=== CÁLCULO DE FACTORES DE ESCALA 2D INDIVIDUALES (ANTEBRAZO = {self.FOREARM_REFERENCE_CM} cm) ===")
        scale_factors = {}
        distance_estimates = {}
        
        for camera_id in ["camera0", "camera1", "camera2"]:
            if cameras_data[camera_id] is not None:
                scale_factor, distance_estimate = self.calculate_scale_factor_from_2d_forearm(
                    cameras_data[camera_id], valid_mask, self.FOREARM_REFERENCE_CM
                )
                scale_factors[camera_id] = scale_factor
                distance_estimates[camera_id] = distance_estimate
                
                if scale_factor is not None and distance_estimate is not None:
                    # Calcular medidas de antebrazo para mostrar detalles
                    coords = cameras_data[camera_id]
                    forearm_details = []
                    
                    if valid_mask[7] and valid_mask[9]:
                        left_forearm = self.calculate_2d_distance(coords[7], coords[9])
                        forearm_details.append(("Izquierdo", left_forearm))
                    
                    if valid_mask[8] and valid_mask[10]:
                        right_forearm = self.calculate_2d_distance(coords[8], coords[10])
                        forearm_details.append(("Derecho", right_forearm))
                    
                    if forearm_details:
                        avg_pixels = np.mean([pixels for _, pixels in forearm_details])
                        print(f"\n{camera_id}:")
                        for side, pixels in forearm_details:
                            print(f"  Antebrazo {side}: {pixels:.1f} píxeles")
                        print(f"  Promedio: {avg_pixels:.1f} píxeles")
                        print(f"  Factor de escala: {scale_factor:.4f} cm/pixel")
                        print(f"  📏 Distancia estimada a la cámara: {distance_estimate:.2f} metros")
                else:
                    print(f"\n{camera_id}: ❌ No se pueden medir antebrazos (puntos no válidos)")
            else:
                scale_factors[camera_id] = None
                distance_estimates[camera_id] = None
                print(f"\n{camera_id}: ❌ Datos no disponibles")
        
        # Análizar medidas para cada cámara con su propio factor de escala
        results = {}
        for camera_id in ["camera0", "camera1", "camera2"]:
            if cameras_data[camera_id] is not None and scale_factors[camera_id] is not None:
                camera_results = self.analyze_2d_measurements_for_camera(
                    cameras_data[camera_id], valid_mask, scale_factors[camera_id], 
                    model_name, camera_id
                )
                results[camera_id] = {
                    "scale_factor": scale_factors[camera_id],
                    "distance_estimate": distance_estimates[camera_id],
                    "measurements": camera_results
                }
            else:
                print(f"\n❌ Saltando {camera_id} (datos no disponibles o factor de escala inválido)")
                results[camera_id] = None
        
        return results

    def run_full_analysis(self):
        """Ejecuta el análisis completo para todos los modelos disponibles."""
        
        logger.info(f"Iniciando análisis comparativo de modelos 2D para {self.patient_id}/{self.session_id}/frame_{self.frame_id}/chunk_{self.chunk_id}")
        
        print("=" * 120)
        print("ANÁLISIS COMPARATIVO DE MODELOS DE POSE ESTIMATION 2D")
        print("=" * 120)
        print(f"Paciente: {self.patient_id}")
        print(f"Sesión: {self.session_id}")
        print(f"Frame: {self.frame_id}")
        print(f"Chunk: {self.chunk_id}")
        print(f"Referencia de antebrazo: {self.FOREARM_REFERENCE_CM} cm")
        print(f"Umbral de confianza: {self.CONFIDENCE_THRESHOLD}")
        print(f"Keypoints analizados: {self.NUM_KEYPOINTS} (de los 133 disponibles)")
        print(f"Parámetros intrínsecos de cámara: fx=418.3, fy=417.5, cx=419.7, cy=264.2")
        print("=" * 120)
        
        # Obtener modelos disponibles
        available_models = self.get_available_models()
        
        if not available_models:
            logger.error("No se encontraron modelos disponibles")
            return
        
        # Ejecutar análisis para cada modelo
        all_results = {}
        
        for model_name in available_models:
            try:
                model_results = self.run_analysis_for_model(model_name)
                all_results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error analizando modelo {model_name}: {e}")
                all_results[model_name] = None
        
        # Generar resumen comparativo final
        self.generate_comparative_summary(all_results)
        
        logger.info("Análisis comparativo de modelos completado exitosamente")

    def generate_comparative_summary(self, all_results: Dict[str, Dict]):
        """Genera un resumen comparativo entre todos los modelos."""
        
        print(f"\n{'#'*120}")
        print("RESUMEN COMPARATIVO ENTRE MODELOS")
        print(f"{'#'*120}")
        
        # Tabla comparativa de realismo por modelo y cámara
        print(f"\n{'Modelo':<15} | {'Camera0':<20} | {'Camera1':<20} | {'Camera2':<20} | {'Promedio':<15}")
        print("-" * 100)
        
        model_averages = []
        
        for model_name, model_results in all_results.items():
            if model_results is None:
                print(f"{model_name:<15} | {'ERROR':<20} | {'ERROR':<20} | {'ERROR':<20} | {'ERROR':<15}")
                continue
            
            camera_percentages = []
            camera_strs = []
            
            for camera_id in ["camera0", "camera1", "camera2"]:
                if camera_id in model_results and model_results[camera_id] is not None:
                    percentage = model_results[camera_id]["measurements"]["realism_percentage"]
                    camera_percentages.append(percentage)
                    
                    if percentage > 80:
                        status = "🟢"
                    elif percentage > 60:
                        status = "🟡"
                    else:
                        status = "🔴"
                    
                    camera_strs.append(f"{status} {percentage:.1f}%")
                else:
                    camera_strs.append("❌ N/A")
            
            # Calcular promedio del modelo
            if camera_percentages:
                model_avg = np.mean(camera_percentages)
                model_averages.append((model_name, model_avg))
                
                if model_avg > 80:
                    avg_status = "🟢"
                elif model_avg > 60:
                    avg_status = "🟡"
                else:
                    avg_status = "🔴"
                
                avg_str = f"{avg_status} {model_avg:.1f}%"
            else:
                avg_str = "❌ N/A"
            
            # Completar con espacios si faltan cámaras
            while len(camera_strs) < 3:
                camera_strs.append("❌ N/A")
            
            print(f"{model_name:<15} | {camera_strs[0]:<20} | {camera_strs[1]:<20} | {camera_strs[2]:<20} | {avg_str:<15}")
        
        # Ranking de modelos
        if model_averages:
            print(f"\n{'='*60}")
            print("RANKING DE MODELOS POR REALISMO ANATÓMICO")
            print(f"{'='*60}")
            
            model_averages.sort(key=lambda x: x[1], reverse=True)
            
            for i, (model_name, avg_percentage) in enumerate(model_averages, 1):
                if avg_percentage > 80:
                    status = "🥇" if i == 1 else "🟢"
                elif avg_percentage > 60:
                    status = "🥈" if i == 2 and len(model_averages) > 1 else "🟡"
                else:
                    status = "🥉" if i == 3 and len(model_averages) > 2 else "🔴"
                
                print(f"{i}. {status} {model_name:<15} - {avg_percentage:.1f}% realismo promedio")
        
        # Detalles de factores de escala y distancias
        print(f"\n{'='*60}")
        print("FACTORES DE ESCALA Y DISTANCIAS POR MODELO Y CÁMARA")
        print(f"{'='*60}")
        
        print(f"{'Modelo':<12} | {'Camera':<8} | {'Escala (cm/px)':<15} | {'Distancia (m)':<15}")
        print("-" * 65)
        
        for model_name, model_results in all_results.items():
            if model_results is None:
                continue
            
            for camera_id in ["camera0", "camera1", "camera2"]:
                if camera_id in model_results and model_results[camera_id] is not None:
                    scale_factor = model_results[camera_id]["scale_factor"]
                    distance_estimate = model_results[camera_id]["distance_estimate"]
                    
                    scale_str = f"{scale_factor:.4f}" if scale_factor else "N/A"
                    dist_str = f"{distance_estimate:.2f}" if distance_estimate else "N/A"
                    
                    camera_num = camera_id[-1]  # Extraer el número de la cámara
                    print(f"{model_name:<12} | {camera_num:<8} | {scale_str:<15} | {dist_str:<15}")
        
        # Resumen de distancias promedio por modelo
        print(f"\n{'='*50}")
        print("DISTANCIA PROMEDIO POR MODELO")
        print(f"{'='*50}")
        
        for model_name, model_results in all_results.items():
            if model_results is None:
                continue
            
            distances = []
            for camera_id in ["camera0", "camera1", "camera2"]:
                if (camera_id in model_results and 
                    model_results[camera_id] is not None and 
                    model_results[camera_id]["distance_estimate"] is not None):
                    distances.append(model_results[camera_id]["distance_estimate"])
            
            if distances:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                print(f"{model_name:<12} | Promedio: {avg_distance:.2f}m ± {std_distance:.2f}m")
            else:
                print(f"{model_name:<12} | Sin datos de distancia válidos")


def main():
    """Función principal de análisis."""
    # Parámetros de ejemplo (ajustar según necesidades)
    patient_id = "patient57"
    session_id = "session57"
    frame_id = 44
    chunk_id = 3
    
    try:
        # Crear analizador y ejecutar
        analyzer = Models2DAnalysis(patient_id, session_id, frame_id, chunk_id)
        analyzer.run_full_analysis()
        
    except Exception as e:
        logger.error(f"Error en análisis principal: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()

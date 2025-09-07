import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Asegurar que config esté en el path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from camera import Camera
from triangulation_svd import triangulate_frame_svd
from triangulation_bundle_adjustment import refine_frame_bundle_adjustment
from reprojection import reprojection_error
from calculate_extrinsics import estimate_extrinsics, print_extrinsic_matrices
from config.camera_intrinsics import CAMERA_INTRINSICS
from full_bundle_adjustment import full_bundle_adjustment, print_extrinsic_matrices_bundle
from backend.tests.reconstruccion_2D import load_ensemble_keypoints

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class GaitAnalysis3D:
    """Análisis completo de reconstrucción 3D"""
    
    # Constantes
    CONFIDENCE_THRESHOLD = 0.5
    FOREARM_REFERENCE_CM = 30.0  # 30 cm como referencia

    def __init__(self, patient_id: str, session_id: str, chunk_id: int, frame_id: int):
        self.patient_id = patient_id
        self.session_id = session_id
        self.chunk_id = chunk_id
        self.frame_id = frame_id
        # Cargar datos reales desde archivos .npy
        self._load_keypoints_data()

    def _load_keypoints_data(self):
        """Carga los datos de keypoints 2D desde los archivos .npy procesados."""
        
        # Construir rutas base - estructura: Server/data/processed/2D_keypoints/patient57/session57/camera0/coordinates/44_3.npy
        base_path = _ROOT / "data" / "processed" / "2D_keypoints" / f"patient{self.patient_id}" / f"session{self.session_id}"

        # Verificar que el directorio base existe
        if not base_path.exists():
            raise FileNotFoundError(f"Directorio base no encontrado: {base_path}")
        
        # Cargar datos para cada cámara
        for camera_id in ["camera0", "camera1", "camera2"]:
            # Construir ruta del archivo - formato: frame_chunk.npy (ej: 44_3.npy)
            coords_file = base_path / camera_id / "coordinates" / f"{self.frame_id}_{self.chunk_id}.npy"
            confs_file = base_path / camera_id / "confidence" / f"{self.frame_id}_{self.chunk_id}.npy"

            # Verificar que ambos archivos existen
            if not coords_file.exists():
                raise FileNotFoundError(f"Archivo de coordenadas no encontrado: {coords_file}")
            if not confs_file.exists():
                raise FileNotFoundError(f"Archivo de confianzas no encontrado: {confs_file}")
            
            try:
                # Cargar coordenadas y confianzas
                coords = np.load(coords_file)
                confs = np.load(confs_file)

                # Asignar a los atributos de la clase según la cámara
                if camera_id == "camera0":
                    self.coordinates_camera_0 = coords
                    self.confidences_camera_0 = confs
                elif camera_id == "camera1":
                    self.coordinates_camera_1 = coords
                    self.confidences_camera_1 = confs
                elif camera_id == "camera2":
                    self.coordinates_camera_2 = coords
                    self.confidences_camera_2 = confs
                    
            except Exception as e:
                raise RuntimeError(f"Error al cargar datos para {camera_id}: {e}")
            
    def get_camera_diagnostics(self, cam, reference_cam):
        """Calcula diagnósticos básicos de una cámara respecto a la de referencia."""
        # Calcular baseline
        baseline = np.linalg.norm(cam.t - reference_cam.t)
        
        # Ángulos de rotación (aproximados)
        rotation_angles = {
            'rx': np.degrees(np.arctan2(cam.R[2,1], cam.R[2,2])),
            'ry': np.degrees(np.arctan2(-cam.R[2,0], np.sqrt(cam.R[2,1]**2 + cam.R[2,2]**2))),
            'rz': np.degrees(np.arctan2(cam.R[1,0], cam.R[0,0]))
        }
        
        return {
            'baseline': baseline,
            'rotation_angles': rotation_angles,
            'translation': cam.t.flatten()
        }
        
    def load_ensemble_keypoints(self, base_data_dir: Path, camera_id: int) -> Dict[int, Dict[str, np.ndarray]]:
        """Carga los keypoints y confianzas del ensemble para la cámara dada"""
        return load_ensemble_keypoints(base_data_dir, self.patient_id, self.session_id, camera_id, self.chunk_id)

    def create_cameras_from_config(self) -> Dict[str, Camera]:
        """Crea las cámaras usando la configuración de intrínsecos"""
        cameras = {}
        for cam_id in ["camera0", "camera1", "camera2"]:
            # Crear cámara base
            cam = Camera.create(cam_id)
            # Los extrínsecos se estimarán posteriormente
            cameras[cam_id] = cam
        return cameras

    def prepare_frame_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepara los datos del frame con los keypoints y confianzas"""
        return {
            "camera0": (self.coordinates_camera_0.copy(), self.confidences_camera_0.copy()),
            "camera1": (self.coordinates_camera_1.copy(), self.confidences_camera_1.copy()),
            "camera2": (self.coordinates_camera_2.copy(), self.confidences_camera_2.copy()),
        }

    #================
    # ANÁLISIS 2D 
    #================
    
    def filter_valid_keypoints(self, confidence_threshold: float = 0.5) -> np.ndarray:
        """Filtra keypoints que tienen confianza > threshold en todas las cámaras."""
        
        # Crear máscara de puntos válidos
        valid_mask = (self.confidences_camera_0 > confidence_threshold) & \
                     (self.confidences_camera_1 > confidence_threshold) & \
                     (self.confidences_camera_2 > confidence_threshold)
        
        print(f"=== FILTRADO DE KEYPOINTS 2D")
        print(f"Umbral de confianza: {confidence_threshold}")
        print(f"Puntos válidos en todas las cámaras: {np.sum(valid_mask)}/{len(valid_mask)}")
        
        return valid_mask

    def calculate_2d_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calcula distancia euclidiana entre dos puntos 2D."""
        return np.linalg.norm(point1 - point2)

    def calculate_scale_factor_from_2d_forearm(self, valid_mask: np.ndarray, real_forearm_length_cm: float = 30.0) -> Dict[str, float]:
        """Calcula factor de escala para cada cámara basado en la longitud del antebrazo en 2D."""
        
        # Índices: 7=Codo_izq, 8=Codo_der, 9=Muñeca_izq, 10=Muñeca_der
        scale_factors = {}
        
        cameras_data = {
            "camera0": self.coordinates_camera_0,
            "camera1": self.coordinates_camera_1, 
            "camera2": self.coordinates_camera_2
        }
        
        print(f"\n=== CÁLCULO DE FACTORES DE ESCALA 2D (ANTEBRAZO = {real_forearm_length_cm} cm)")
        
        for cam_name, coords in cameras_data.items():
            forearm_measurements = []
            
            # Antebrazo izquierdo (codo_izq a muñeca_izq)
            if valid_mask[7] and valid_mask[9]:  # Codo_izq y Muñeca_izq válidos
                left_forearm_2d = self.calculate_2d_distance(coords[7], coords[9])
                forearm_measurements.append(("Izquierdo", left_forearm_2d))
            
            # Antebrazo derecho (codo_der a muñeca_der)
            if valid_mask[8] and valid_mask[10]:  # Codo_der y Muñeca_der válidos
                right_forearm_2d = self.calculate_2d_distance(coords[8], coords[10])
                forearm_measurements.append(("Derecho", right_forearm_2d))
            
            if forearm_measurements:
                # Usar promedio de las medidas disponibles
                avg_forearm_pixels = np.mean([pixels for _, pixels in forearm_measurements])
                scale_factor = real_forearm_length_cm / avg_forearm_pixels  # cm/pixel
                scale_factors[cam_name] = scale_factor
                
                print(f"\n{cam_name}:")
                for side, pixels in forearm_measurements:
                    print(f"  Antebrazo {side}: {pixels:.1f} píxeles")
                print(f"  Promedio: {avg_forearm_pixels:.1f} píxeles")
                print(f"  Factor de escala: {scale_factor:.4f} cm/pixel")
            else:
                scale_factors[cam_name] = None
                print(f"\n{cam_name}: No se pueden medir antebrazos (puntos no válidos)")
        
        return scale_factors

    def analyze_2d_body_measurements(self, valid_mask: np.ndarray, scale_factors: Dict[str, float]):
        """Analiza medidas corporales 2D escaladas para cada cámara"""
        
        cameras_data = {
            "camera0": self.coordinates_camera_0,
            "camera1": self.coordinates_camera_1, 
            "camera2": self.coordinates_camera_2
        }
        
        def distance_2d_scaled(coords: np.ndarray, p1_idx: int, p2_idx: int, scale_factor: float) -> float:
            """Calcula distancia 2D escalada entre dos keypoints."""
            if not (valid_mask[p1_idx] and valid_mask[p2_idx]) or scale_factor is None:
                return np.nan
            pixel_distance = self.calculate_2d_distance(coords[p1_idx], coords[p2_idx])
            return pixel_distance * scale_factor  # convertir a cm
        
        # Definir medidas a calcular
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
            ("Estatura aprox (nariz-tobillo_izq)", 0, 15, "150-190 cm"),
            ("Estatura aprox (nariz-tobillo_der)", 0, 16, "150-190 cm"),
            ("Envergadura (muñeca_izq - muñeca_der)", 9, 10, "150-180 cm"),
        ]
        
        # Calcular factor de escala común (media de todas las cámaras)
        valid_factors = [f for f in scale_factors.values() if f is not None]
        if valid_factors:
            common_scale_factor = np.mean(valid_factors)
            std_scale = np.std(valid_factors)
            print(f"\nFACTOR DE ESCALA COMÚN CALCULADO:")
            print(f"Factor promedio: {common_scale_factor:.4f} cm/pixel (de {len(valid_factors)} cámaras)")
            print(f"Desviación estándar: {std_scale:.4f} cm/pixel")
        else:
            common_scale_factor = None
            print(f"\nNo se pudo calcular factor de escala común")
            
        for cam_name, coords in cameras_data.items():
            if scale_factors[cam_name] is None:
                print(f"\nSaltando {cam_name} (no se pudo calcular factor de escala)")
                continue
                
            print(f"\n{'='*80}")
            print(f"ANÁLISIS DE MEDIDAS 2D ESCALADAS - {cam_name.upper()}")
            print(f"Factor de escala: {common_scale_factor:.4f} cm/pixel")
            print(f"{'='*80}")
            
            print(f"{'Medida':<40} | {'Valor':<12} | {'Rango Normal':<15} | {'Estado'}")
            print("-" * 85)
            
            valid_measurements = 0
            realistic_measurements = 0
            
            for name, p1_idx, p2_idx, normal_range in measurement_definitions:
                value_cm = distance_2d_scaled(coords, p1_idx, p2_idx, common_scale_factor)
                
                if np.isnan(value_cm):
                    status = "N/A"
                    value_str = "N/A"
                else:
                    value_str = f"{value_cm:.1f} cm"
                    
                    # Análisis básico de realismo
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
                        
                    status = "OK" if realistic else "Fuera rango"
                    if realistic:
                        realistic_measurements += 1
                    valid_measurements += 1
                
                print(f"{name:<40} | {value_str:<12} | {normal_range:<15} | {status}")
            

    #===============
    # ANÁLISIS 3D
    #===============
    
    def calculate_scale_factor_from_forearm(self, points_3d: np.ndarray, real_forearm_length_cm: float = 30.0):
        """Calcula el factor de escala basado en la longitud real del antebrazo"""
        
        # Índices para codo y muñeca (derecha e izquierda)
        # 7: Codo_izq, 8: Codo_der, 9: Muñeca_izq, 10: Muñeca_der
        forearm_measurements = []
        
        # Antebrazo izquierdo (codo_izq a muñeca_izq)
        if not np.isnan(points_3d[7, 0]) and not np.isnan(points_3d[9, 0]):
            left_forearm = np.linalg.norm(points_3d[7] - points_3d[9])
            forearm_measurements.append(("Antebrazo izquierdo", left_forearm))
        
        # Antebrazo derecho (codo_der a muñeca_der)
        if not np.isnan(points_3d[8, 0]) and not np.isnan(points_3d[10, 0]):
            right_forearm = np.linalg.norm(points_3d[8] - points_3d[10])
            forearm_measurements.append(("Antebrazo derecho", right_forearm))
        
        if not forearm_measurements:
            print("ERROR: No se pueden calcular medidas de antebrazo")
            return None
        
        # Usar el promedio de las medidas disponibles
        avg_forearm_length_m = np.mean([length for _, length in forearm_measurements])
        real_forearm_length_m = real_forearm_length_cm / 100.0  # convertir a metros
        
        # Factor de escala
        scale_factor = real_forearm_length_m / avg_forearm_length_m
        
        print(f"\n=== CÁLCULO DE FACTOR DE ESCALA (BASADO EN ANTEBRAZO)")
        print(f"Longitud real del antebrazo: {real_forearm_length_cm:.1f} cm")
        for name, length in forearm_measurements:
            print(f"{name}: {length*100:.1f} cm (3D estimado)")
        print(f"Longitud promedio estimada: {avg_forearm_length_m*100:.1f} cm")
        print(f"Factor de escala calculado: {scale_factor:.4f}")
        
        return scale_factor

    def analyze_body_measurements_scaled(self, points_3d: np.ndarray, method_name: str, scale_factor: float):
        """Analiza las medidas corporales con escala corregida"""
        
        # Aplicar factor de escala a todos los puntos
        scaled_points = points_3d * scale_factor
        
        def distance_3d(p1_idx: int, p2_idx: int) -> float:
            if np.isnan(scaled_points[p1_idx, 0]) or np.isnan(scaled_points[p2_idx, 0]):
                return np.nan
            return np.linalg.norm(scaled_points[p1_idx] - scaled_points[p2_idx]) * 100  # convertir a cm
        
        print(f"\n{'='*70}")
        print(f"ANÁLISIS DE MEDIDAS CORPORALES ESCALADAS ({method_name})")
        print(f"Referencia: Antebrazo = 30.0 cm, Factor de escala: {scale_factor:.4f}")
        print(f"{'='*70}")

        # Medidas principales del cuerpo
        measurements = []
        
        # Cabeza y cuello
        measurements.extend([
            ("Ancho cara (ojo_izq - ojo_der)", distance_3d(1, 2), "7-10 cm"),
            ("Distancia ojos-nariz promedio", 
             np.nanmean([distance_3d(0, 1), distance_3d(0, 2)]), "2-4 cm"),
        ])
        
        # Torso
        measurements.extend([
            ("Ancho hombros", distance_3d(5, 6), "35-45 cm"),
            ("Alto torso (hombro_izq - cadera_izq)", distance_3d(5, 11), "50-70 cm"),
            ("Alto torso (hombro_der - cadera_der)", distance_3d(6, 12), "50-70 cm"),
            ("Ancho caderas", distance_3d(11, 12), "25-35 cm"),
        ])
        
        # Brazo izquierdo
        measurements.extend([
            ("Brazo izq (hombro-codo)", distance_3d(5, 7), "28-36 cm"),
            ("Antebrazo izq (codo-muñeca)", distance_3d(7, 9), "23-30 cm"),
            ("Brazo completo izq (hombro-muñeca)", distance_3d(5, 9), "55-70 cm"),
        ])
        
        # Brazo derecho
        measurements.extend([
            ("Brazo der (hombro-codo)", distance_3d(6, 8), "28-36 cm"),
            ("Antebrazo der (codo-muñeca)", distance_3d(8, 10), "23-30 cm"),
            ("Brazo completo der (hombro-muñeca)", distance_3d(6, 10), "55-70 cm"),
        ])
        
        # Pierna izquierda
        measurements.extend([
            ("Muslo izq (cadera-rodilla)", distance_3d(11, 13), "35-50 cm"),
            ("Pantorrilla izq (rodilla-tobillo)", distance_3d(13, 15), "35-45 cm"),
            ("Pierna completa izq (cadera-tobillo)", distance_3d(11, 15), "75-100 cm"),
        ])
        
        # Pierna derecha
        measurements.extend([
            ("Muslo der (cadera-rodilla)", distance_3d(12, 14), "35-50 cm"),
            ("Pantorrilla der (rodilla-tobillo)", distance_3d(14, 16), "35-45 cm"),
            ("Pierna completa der (cadera-tobillo)", distance_3d(12, 16), "75-100 cm"),
        ])
        
        # Medidas adicionales
        measurements.extend([
            ("Estatura aprox (nariz-tobillo_izq)", 
             distance_3d(0, 15) if not np.isnan(distance_3d(0, 15)) else np.nan, "150-190 cm"),
            ("Estatura aprox (nariz-tobillo_der)", 
             distance_3d(0, 16) if not np.isnan(distance_3d(0, 16)) else np.nan, "150-190 cm"),
            ("Envergadura (muñeca_izq - muñeca_der)", distance_3d(9, 10), "150-180 cm"),
        ])
        
        # Mostrar resultados
        print(f"{'Medida':<40} | {'Valor':<12} | {'Rango Normal':<15} | {'Estado'}")
        print("-" * 85)
        
        valid_measurements = 0
        realistic_measurements = 0
        
        for name, value, normal_range in measurements:
            if np.isnan(value):
                status = "N/A"
                value_str = "N/A"
            else:
                value_str = f"{value:.1f} cm"
                
                # Análisis de realismo basado en rangos
                if "cara" in name.lower():
                    realistic = 5 <= value <= 15
                elif "ojos-nariz" in name.lower():
                    realistic = 1 <= value <= 6
                elif "hombros" in name.lower():
                    realistic = 25 <= value <= 55
                elif "torso" in name.lower():
                    realistic = 40 <= value <= 80
                elif "caderas" in name.lower():
                    realistic = 20 <= value <= 45
                elif "brazo" in name.lower() and "completo" not in name.lower():
                    realistic = 20 <= value <= 45
                elif "brazo completo" in name.lower():
                    realistic = 45 <= value <= 80
                elif "muslo" in name.lower():
                    realistic = 30 <= value <= 60
                elif "pantorrilla" in name.lower():
                    realistic = 30 <= value <= 55
                elif "pierna completa" in name.lower():
                    realistic = 65 <= value <= 110
                elif "estatura" in name.lower():
                    realistic = 140 <= value <= 200
                elif "envergadura" in name.lower():
                    realistic = 140 <= value <= 200
                else:
                    realistic = True
                    
                status = "OK" if realistic else "Fuera rango"
                if realistic:
                    realistic_measurements += 1
                valid_measurements += 1
            
            print(f"{name:<40} | {value_str:<12} | {normal_range:<15} | {status}")

    def run_full_analysis(self):
        """Ejecuta el análisis completo"""

        logger.info(f"Iniciando análisis completo para patient{self.patient_id}/session{self.session_id}/chunk_{self.chunk_id}/frame_{self.frame_id}")

        print("=== ANÁLISIS DE MEDIDAS CORPORALES 2D")
        print("Basado en keypoints 2D con antebrazo de referencia = 30.0 cm")
        print()
        
        # PARTE 1: ANÁLISIS 2D
        
        # Filtrar keypoints válidos
        valid_mask = self.filter_valid_keypoints(confidence_threshold=self.CONFIDENCE_THRESHOLD)
        
        # Calcular factores de escala por cámara
        scale_factors = self.calculate_scale_factor_from_2d_forearm(valid_mask, real_forearm_length_cm=30.0)
        
        # Análisis de medidas corporales por cámara
        self.analyze_2d_body_measurements(valid_mask, scale_factors)
        
        # PARTE 2: ANÁLISIS 3D
        
        logger.info("Configurando cámaras...")
        # Fijar semilla para reproducibilidad
        np.random.seed(42)
        
        # Preparar datos
        cameras = self.create_cameras_from_config()
        frame_keypoints = self.prepare_frame_data()
        
        logger.info("Estimando parámetros extrínsecos...")
        
        # Método Riguroso: Estimación con geometría epipolar
        print("\n=== Estimación Rigurosa con Geometría Epipolar")
        try:
            cameras_rigorous = estimate_extrinsics(
                cameras, frame_keypoints, self.CONFIDENCE_THRESHOLD
            )
            
            # Mostrar matrices de parámetros extrínsecos estimados
            print_extrinsic_matrices(cameras_rigorous, "PARÁMETROS EXTRÍNSECOS ESTIMADOS")
            
            logger.info("Ejecutando triangulación 3D...")
            
            # PARTE 1: Triangulación SVD
            print(f"\n{'='*50}")
            print("PARTE 1: TRIANGULACIÓN SVD (Sin refinamiento)")
            print(f"{'='*50}")
            points_3d_svd = triangulate_frame_svd(cameras_rigorous, frame_keypoints, self.CONFIDENCE_THRESHOLD)
            
            svd_count = np.sum(~np.isnan(points_3d_svd[:, 0]))
            print(f"Triangulación SVD: {svd_count}/{len(points_3d_svd)} puntos válidos")
            
            # Errores de reproyección con SVD
            errors_svd = reprojection_error(points_3d_svd, cameras_rigorous, frame_keypoints)
            print(f"\nErrores de reproyección (SVD):")
            for cam_id, error in errors_svd.items():
                print(f"  {cam_id}: {error:.2f} píxeles")
            
            # PARTE 2: Bundle Adjustment
            print(f"\n{'='*50}")
            print("PARTE 2: BUNDLE ADJUSTMENT (Refinamiento)")
            print(f"{'='*50}")
            
            if svd_count > 0:
                points_3d_ba = refine_frame_bundle_adjustment(
                    points_3d_svd, cameras_rigorous, frame_keypoints
                )
                
                ba_count = np.sum(~np.isnan(points_3d_ba[:, 0]))
                print(f"Bundle Adjustment: {svd_count} -> {ba_count} puntos válidos")
                
                # Errores de reproyección con Bundle Adjustment
                errors_ba = reprojection_error(points_3d_ba, cameras_rigorous, frame_keypoints)
                print(f"\nErrores de reproyección (Bundle Adjustment):")
                for cam_id, error in errors_ba.items():
                    print(f"  {cam_id}: {error:.2f} píxeles")
                
                # COMPARACIÓN
                print(f"\n{'='*50}")
                print("COMPARACIÓN SVD vs BUNDLE ADJUSTMENT")
                print(f"{'='*50}")
                
                print("Mejora en errores de reproyección:")
                for cam_id in errors_svd.keys():
                    improvement = errors_svd[cam_id] - errors_ba[cam_id]
                    print(f"  {cam_id}: {errors_svd[cam_id]:.2f} -> {errors_ba[cam_id]:.2f} px ({improvement:+.2f} px)")
                
                avg_error_svd = np.mean(list(errors_svd.values()))
                avg_error_ba = np.mean(list(errors_ba.values()))
                total_improvement = avg_error_svd - avg_error_ba
                print(f"\nError promedio: {avg_error_svd:.2f} -> {avg_error_ba:.2f} px ({total_improvement:+.2f} px)")
                
            else:
                print("ERROR: No hay puntos válidos de SVD para refinar con Bundle Adjustment")
                points_3d_ba = points_3d_svd
            
            # PARTE 3: Full Bundle Adjustment
            print(f"\n{'='*50}")
            print("PARTE 3: FULL BUNDLE ADJUSTMENT (Optimización Completa)")
            print(f"{'='*50}")
            
            if svd_count > 0:
                try:
                    # Preparar datos para full bundle adjustment
                    points_3d_full_ba, cameras_full_ba = full_bundle_adjustment(
                        points_3d_svd, cameras_rigorous, frame_keypoints, 
                        confidence_threshold=self.CONFIDENCE_THRESHOLD
                    )
                    
                    full_ba_count = np.sum(~np.isnan(points_3d_full_ba[:, 0]))
                    print(f"Full Bundle Adjustment: {svd_count} -> {full_ba_count} puntos válidos")
                    
                    # Errores de reproyección con Full Bundle Adjustment
                    errors_full_ba = reprojection_error(points_3d_full_ba, cameras_full_ba, frame_keypoints)
                    print(f"\nErrores de reproyección (Full Bundle Adjustment):")
                    for cam_id, error in errors_full_ba.items():
                        print(f"  {cam_id}: {error:.2f} píxeles")
                    
                    # Mostrar matrices optimizadas por Full Bundle Adjustment
                    print_extrinsic_matrices_bundle(cameras_full_ba, "PARÁMETROS EXTRÍNSECOS OPTIMIZADOS - FULL BUNDLE ADJUSTMENT")
                    
                    # COMPARACIÓN COMPLETA
                    print(f"\n{'='*60}")
                    print("COMPARACIÓN SVD vs BA vs FULL BA")
                    print(f"{'='*60}")
                    
                    print("Mejora en errores de reproyección:")
                    for cam_id in errors_svd.keys():
                        svd_error = errors_svd[cam_id]
                        ba_error = errors_ba[cam_id]
                        full_ba_error = errors_full_ba[cam_id]
                        
                        ba_improvement = svd_error - ba_error
                        full_ba_improvement = svd_error - full_ba_error
                        
                        print(f"  {cam_id}:")
                        print(f"    SVD: {svd_error:.2f} px")
                        print(f"    BA:  {ba_error:.2f} px ({ba_improvement:+.2f})")
                        print(f"    Full BA: {full_ba_error:.2f} px ({full_ba_improvement:+.2f})")
                    
                    avg_error_svd = np.mean(list(errors_svd.values()))
                    avg_error_ba = np.mean(list(errors_ba.values()))
                    avg_error_full_ba = np.mean(list(errors_full_ba.values()))
                    
                    ba_total_improvement = avg_error_svd - avg_error_ba
                    full_ba_total_improvement = avg_error_svd - avg_error_full_ba
                    
                    print(f"\nError promedio:")
                    print(f"  SVD: {avg_error_svd:.2f} px")
                    print(f"  BA:  {avg_error_ba:.2f} px ({ba_total_improvement:+.2f})")
                    print(f"  Full BA: {avg_error_full_ba:.2f} px ({full_ba_total_improvement:+.2f})")
                    
                except Exception as e:
                    logger.error(f"Error en Full Bundle Adjustment: {e}")
                    print(f"ERROR en Full Bundle Adjustment: {e}")
                    # Usar los resultados de BA normal como fallback
                    cameras_full_ba = cameras_rigorous
                    points_3d_full_ba = points_3d_ba
            else:
                print("ERROR: No hay puntos válidos de SVD para Full Bundle Adjustment")
                cameras_full_ba = cameras_rigorous
                points_3d_full_ba = points_3d_svd
            
            logger.info(f"Triangulación completada - SVD: {len(points_3d_svd)} puntos, BA: {len(points_3d_ba)} puntos, Full BA: {len(points_3d_full_ba)} puntos")
            
            # ANÁLISIS DE MEDIDAS CORPORALES ESCALADAS
            
            methods_data = [
                ("SVD", points_3d_svd), 
                ("Bundle_Adjustment", points_3d_ba),
                ("Full_Bundle_Adjustment", points_3d_full_ba)
            ]
            
            for method_name, points_3d in methods_data:
                # Calcular factor de escala basado en antebrazo
                scale_factor = self.calculate_scale_factor_from_forearm(points_3d, 30.0)
                
                if scale_factor is not None:
                    # Análisis con escala corregida
                    self.analyze_body_measurements_scaled(points_3d, method_name, scale_factor)
                else:
                    print(f"ERROR: No se pudo calcular factor de escala para {method_name}")
            
            logger.info("Análisis completo finalizado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en estimación de extrínsecos: {e}")
            print(f"ERROR en reconstrucción 3D: {e}")


def main():
    patient_id = 57
    session_id = 57
    chunk_id = 6
    frame_id = 44
    
    # Crear analizador y ejecutar
    analyzer = GaitAnalysis3D(patient_id, session_id, chunk_id, frame_id)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
"""
Validación de reconstrucción 3D mediante reproyección y análisis de errores.
Compara puntos 3D proyectados con keypoints 2D originales.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import matplotlib.pyplot as plt
from .camera import Camera

logger = logging.getLogger(__name__)


def validate_reprojection(
    cameras: Dict[str, Camera],
    points_3d: np.ndarray,
    keypoints_2d: Dict[str, np.ndarray],
    error_threshold: float = 2.0,
    save_plots: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Valida la calidad de reconstrucción 3D mediante reproyección.
    
    Proyecta los puntos 3D reconstruidos a cada cámara y compara con
    los keypoints 2D originales para calcular errores de reproyección.
    
    Args:
        cameras: Dict con objetos Camera configurados con extrínsecos
        points_3d: Array Nx3 con puntos 3D reconstruidos
        keypoints_2d: Dict con keypoints 2D originales por cámara
        error_threshold: Umbral para considerar un punto como válido (píxeles)
        save_plots: Si guardar gráficos de análisis de errores
        output_dir: Directorio para guardar plots (opcional)
        
    Returns:
        Dict con estadísticas de validación por cámara y globales
    """
    
    if len(points_3d) == 0:
        return {"status": "no_points", "cameras": {}}
    
    validation_results = {
        "cameras": {},
        "global_stats": {},
        "status": "success"
    }
    
    all_errors = []
    total_valid_points = 0
    total_points_tested = 0
    
    for camera_id, camera in cameras.items():
        if camera_id not in keypoints_2d:
            continue
            
        kp_2d_original = keypoints_2d[camera_id]
        num_points = min(len(points_3d), len(kp_2d_original))
        
        if num_points == 0:
            continue
        
        # Proyectar puntos 3D a esta cámara
        points_3d_subset = points_3d[:num_points]
        projected_2d = camera.project_points(points_3d_subset)
        
        # Obtener keypoints originales correspondientes
        original_2d = kp_2d_original[:num_points]
        
        # Filtrar puntos válidos (sin NaN en originales)
        valid_mask = ~np.any(np.isnan(original_2d), axis=1)
        
        if np.sum(valid_mask) == 0:
            validation_results["cameras"][camera_id] = {
                "status": "no_valid_points",
                "num_points": 0
            }
            continue
        
        projected_valid = projected_2d[valid_mask]
        original_valid = original_2d[valid_mask]
        
        # Calcular errores de reproyección
        reprojection_errors = np.linalg.norm(projected_valid - original_valid, axis=1)
        
        # Estadísticas para esta cámara
        mean_error = np.mean(reprojection_errors)
        median_error = np.median(reprojection_errors)
        std_error = np.std(reprojection_errors)
        max_error = np.max(reprojection_errors)
        min_error = np.min(reprojection_errors)
        
        # Puntos dentro del umbral
        valid_points = np.sum(reprojection_errors <= error_threshold)
        valid_percentage = (valid_points / len(reprojection_errors)) * 100
        
        validation_results["cameras"][camera_id] = {
            "num_points": len(reprojection_errors),
            "mean_error": float(mean_error),
            "median_error": float(median_error),
            "std_error": float(std_error),
            "max_error": float(max_error),
            "min_error": float(min_error),
            "valid_points": int(valid_points),
            "valid_percentage": float(valid_percentage),
            "errors": reprojection_errors.tolist()
        }
        
        # Acumular para estadísticas globales
        all_errors.extend(reprojection_errors)
        total_valid_points += valid_points
        total_points_tested += len(reprojection_errors)
        
        logger.info(f"Cámara {camera_id}: Error promedio {mean_error:.2f}px, "
                   f"{valid_percentage:.1f}% puntos válidos")
    
    # Estadísticas globales
    if all_errors:
        all_errors = np.array(all_errors)
        
        validation_results["global_stats"] = {
            "total_points": total_points_tested,
            "valid_points": total_valid_points,
            "valid_percentage": (total_valid_points / total_points_tested) * 100,
            "mean_error": float(np.mean(all_errors)),
            "median_error": float(np.median(all_errors)),
            "std_error": float(np.std(all_errors)),
            "max_error": float(np.max(all_errors)),
            "min_error": float(np.min(all_errors)),
            "percentile_95": float(np.percentile(all_errors, 95)),
            "error_threshold": error_threshold
        }
        
        logger.info(f"Validación global: {total_valid_points}/{total_points_tested} puntos válidos "
                   f"({validation_results['global_stats']['valid_percentage']:.1f}%), "
                   f"error promedio: {validation_results['global_stats']['mean_error']:.2f}px")
    
    # Generar plots si se solicita
    if save_plots and all_errors:
        _generate_validation_plots(validation_results, output_dir)
    
    return validation_results


def _generate_validation_plots(validation_results: Dict, output_dir: Optional[str] = None):
    """
    Genera gráficos de análisis de errores de reproyección.
    
    Args:
        validation_results: Resultados de validación
        output_dir: Directorio para guardar plots
    """
    
    try:
        import matplotlib.pyplot as plt
        import os
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Análisis de Errores de Reproyección', fontsize=16)
        
        # Recopilar todos los errores por cámara
        all_errors_by_camera = {}
        all_errors_combined = []
        
        for camera_id, stats in validation_results["cameras"].items():
            if "errors" in stats:
                errors = np.array(stats["errors"])
                all_errors_by_camera[camera_id] = errors
                all_errors_combined.extend(errors)
        
        all_errors_combined = np.array(all_errors_combined)
        
        # 1. Histograma de errores por cámara
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (camera_id, errors) in enumerate(all_errors_by_camera.items()):
            ax1.hist(errors, bins=30, alpha=0.7, label=camera_id, 
                    color=colors[i % len(colors)], density=True)
        ax1.set_xlabel('Error de Reproyección (píxeles)')
        ax1.set_ylabel('Densidad')
        ax1.set_title('Distribución de Errores por Cámara')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Boxplot de errores por cámara
        ax2 = axes[0, 1]
        errors_list = [errors for errors in all_errors_by_camera.values()]
        camera_labels = list(all_errors_by_camera.keys())
        ax2.boxplot(errors_list, labels=camera_labels)
        ax2.set_ylabel('Error de Reproyección (píxeles)')
        ax2.set_title('Distribución de Errores por Cámara')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histograma global
        ax3 = axes[1, 0]
        ax3.hist(all_errors_combined, bins=50, alpha=0.7, color='skyblue', density=True)
        
        # Agregar líneas de estadísticas
        mean_err = validation_results["global_stats"]["mean_error"]
        median_err = validation_results["global_stats"]["median_error"]
        p95_err = validation_results["global_stats"]["percentile_95"]
        
        ax3.axvline(mean_err, color='red', linestyle='--', label=f'Media: {mean_err:.2f}px')
        ax3.axvline(median_err, color='green', linestyle='--', label=f'Mediana: {median_err:.2f}px')
        ax3.axvline(p95_err, color='orange', linestyle='--', label=f'P95: {p95_err:.2f}px')
        
        ax3.set_xlabel('Error de Reproyección (píxeles)')
        ax3.set_ylabel('Densidad')
        ax3.set_title('Distribución Global de Errores')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Estadísticas resumidas
        ax4 = axes[1, 1]
        ax4.axis('off')  # Ocultar ejes
        
        # Texto con estadísticas
        stats_text = f"""
        ESTADÍSTICAS GLOBALES
        
        Puntos totales: {validation_results["global_stats"]["total_points"]}
        Puntos válidos: {validation_results["global_stats"]["valid_points"]}
        Porcentaje válido: {validation_results["global_stats"]["valid_percentage"]:.1f}%
        
        Error promedio: {validation_results["global_stats"]["mean_error"]:.2f} px
        Error mediano: {validation_results["global_stats"]["median_error"]:.2f} px
        Desviación estándar: {validation_results["global_stats"]["std_error"]:.2f} px
        Error máximo: {validation_results["global_stats"]["max_error"]:.2f} px
        Percentil 95: {validation_results["global_stats"]["percentile_95"]:.2f} px
        
        Umbral usado: {validation_results["global_stats"]["error_threshold"]:.1f} px
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar plot si se especifica directorio
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'reprojection_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot guardado en: {plot_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib no disponible, saltando generación de plots")
    except Exception as e:
        logger.error(f"Error generando plots: {e}")


def calculate_reconstruction_quality_score(validation_results: Dict) -> float:
    """
    Calcula un score de calidad de la reconstrucción (0-100).
    
    Basado en el porcentaje de puntos válidos y el error promedio de reproyección.
    
    Args:
        validation_results: Resultados de validate_reprojection
        
    Returns:
        Score de calidad entre 0 (pésimo) y 100 (perfecto)
    """
    
    if "global_stats" not in validation_results:
        return 0.0
    
    stats = validation_results["global_stats"]
    
    # Componente 1: Porcentaje de puntos válidos (0-50 puntos)
    valid_percentage = stats.get("valid_percentage", 0)
    valid_score = min(50, valid_percentage * 0.5)
    
    # Componente 2: Error promedio (0-50 puntos, inversamente proporcional)
    mean_error = stats.get("mean_error", 10)
    threshold = stats.get("error_threshold", 2.0)
    
    # Score basado en error: 50 puntos si error=0, 0 puntos si error>=threshold*2
    error_score = max(0, 50 * (1 - mean_error / (threshold * 2)))
    
    total_score = valid_score + error_score
    
    logger.info(f"Score de calidad: {total_score:.1f}/100 "
               f"(válidos: {valid_score:.1f}, error: {error_score:.1f})")
    
    return total_score


def filter_outlier_points_3d(
    points_3d: np.ndarray,
    validation_results: Dict,
    error_threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtra puntos 3D que son outliers según errores de reproyección.
    
    Args:
        points_3d: Array Nx3 con puntos 3D originales
        validation_results: Resultados de validate_reprojection
        error_threshold: Umbral de error para considerar outlier
        
    Returns:
        Tupla (points_3d_filtered, valid_indices):
        - points_3d_filtered: Puntos 3D sin outliers
        - valid_indices: Índices de puntos válidos en array original
    """
    
    if len(points_3d) == 0:
        return points_3d, np.array([], dtype=int)
    
    # Recopilar errores promedio por punto desde todas las cámaras
    num_points = len(points_3d)
    point_errors = np.full(num_points, np.inf)
    point_counts = np.zeros(num_points)
    
    for camera_id, stats in validation_results["cameras"].items():
        if "errors" not in stats:
            continue
            
        errors = np.array(stats["errors"])
        num_camera_points = min(len(errors), num_points)
        
        for i in range(num_camera_points):
            if point_errors[i] == np.inf:
                point_errors[i] = errors[i]
                point_counts[i] = 1
            else:
                # Promedio acumulativo
                point_errors[i] = (point_errors[i] * point_counts[i] + errors[i]) / (point_counts[i] + 1)
                point_counts[i] += 1
    
    # Filtrar puntos con error bajo el umbral
    valid_mask = (point_errors <= error_threshold) & (point_counts > 0)
    valid_indices = np.where(valid_mask)[0]
    points_3d_filtered = points_3d[valid_mask]
    
    logger.info(f"Filtrado de outliers: {len(points_3d_filtered)}/{len(points_3d)} puntos mantenidos")
    
    return points_3d_filtered, valid_indices

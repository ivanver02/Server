"""
Test Runner Principal - Sistema de Análisis de Marcha

Este script ejecuta toda la suite de tests del sistema de forma organizada:
- Tests de MMPose y ensemble learning
- Tests de calibración de cámaras  
- Tests de triangulación 3D
- Tests de reconstrucción completa
- Tests de integración del pipeline

Uso:
    python -m backend.tests.run_all_tests
    python -m backend.tests.run_all_tests --quick     # Solo tests rápidos
    python -m backend.tests.run_all_tests --module mmpose  # Solo un módulo
"""

import sys
import time
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Imprimir banner del sistema de tests"""
    print("🦴" + "=" * 68 + "🦴")
    print("🧪 SISTEMA DE TESTS - ANÁLISIS DE MARCHA PARA GONARTROSIS 🧪")
    print("🦴" + "=" * 68 + "🦴")
    print()

def print_section(title: str):
    """Imprimir título de sección"""
    print(f"\n{'='*60}")
    print(f"🔬 {title}")
    print(f"{'='*60}")

def run_quick_tests() -> Dict[str, bool]:
    """Ejecutar tests rápidos de todos los módulos"""
    print_banner()
    print("⚡ EJECUTANDO TESTS RÁPIDOS")
    print("Estos tests validan funcionalidad básica sin dependencias pesadas")
    print()
    
    results = {}
    
    # Test 1: MMPose básico
    print_section("Test Rápido MMPose")
    try:
        from .test_mmpose import run_quick_mmpose_test
        results['mmpose'] = run_quick_mmpose_test()
    except Exception as e:
        print(f"❌ Error en test MMPose: {e}")
        results['mmpose'] = False
    
    # Test 2: Calibración básica
    print_section("Test Rápido Calibración")
    try:
        from .test_calibration import test_quick_calibration
        results['calibration'] = test_quick_calibration()
    except Exception as e:
        print(f"❌ Error en test calibración: {e}")
        results['calibration'] = False
    
    # Test 3: Triangulación básica
    print_section("Test Rápido Triangulación")
    try:
        from .test_triangulation import test_quick_triangulation
        results['triangulation'] = test_quick_triangulation()
    except Exception as e:
        print(f"❌ Error en test triangulación: {e}")
        results['triangulation'] = False
    
    # Test 4: Reconstrucción básica
    print_section("Test Rápido Reconstrucción")
    try:
        from .test_reconstruction import run_quick_test
        results['reconstruction'] = run_quick_test()
    except Exception as e:
        print(f"❌ Error en test reconstrucción: {e}")
        results['reconstruction'] = False
    
    return results

def run_full_tests() -> Dict[str, bool]:
    """Ejecutar suite completa de tests"""
    print_banner()
    print("🔬 EJECUTANDO SUITE COMPLETA DE TESTS")
    print("Incluye tests con dependencias y validación exhaustiva")
    print()
    
    results = {}
    
    # Test 1: MMPose completo
    print_section("Tests MMPose Completos")
    try:
        from .test_mmpose import run_mmpose_tests
        results['mmpose'] = run_mmpose_tests()
    except Exception as e:
        print(f"❌ Error en tests MMPose: {e}")
        results['mmpose'] = False
    
    # Test 2: Calibración completa
    print_section("Tests Calibración Completos")
    try:
        from .test_calibration import run_calibration_tests
        results['calibration'] = run_calibration_tests()
    except Exception as e:
        print(f"❌ Error en tests calibración: {e}")
        results['calibration'] = False
    
    # Test 3: Triangulación completa
    print_section("Tests Triangulación Completos")
    try:
        from .test_triangulation import run_triangulation_tests
        results['triangulation'] = run_triangulation_tests()
    except Exception as e:
        print(f"❌ Error en tests triangulación: {e}")
        results['triangulation'] = False
    
    # Test 4: Reconstrucción completa
    print_section("Tests Reconstrucción Completos")
    try:
        from .test_reconstruction import ReconstructionTester
        tester = ReconstructionTester()
        
        # Test datos sintéticos
        print("🎲 Testing datos sintéticos...")
        synthetic_result = tester.test_synthetic_data()
        
        # Test pipeline completo
        print("🔄 Testing pipeline completo...")
        pipeline_result = tester.test_full_pipeline()
        
        results['reconstruction'] = synthetic_result and pipeline_result
        
    except Exception as e:
        print(f"❌ Error en tests reconstrucción: {e}")
        results['reconstruction'] = False
    
    return results

def run_module_tests(module_name: str) -> bool:
    """Ejecutar tests de un módulo específico"""
    print_banner()
    print(f"🎯 EJECUTANDO TESTS DEL MÓDULO: {module_name.upper()}")
    print()
    
    try:
        if module_name == 'mmpose':
            from .test_mmpose import run_mmpose_tests
            return run_mmpose_tests()
        
        elif module_name == 'calibration':
            from .test_calibration import run_calibration_tests
            return run_calibration_tests()
        
        elif module_name == 'triangulation':
            from .test_triangulation import run_triangulation_tests
            return run_triangulation_tests()
        
        elif module_name == 'reconstruction':
            from .test_reconstruction import ReconstructionTester
            tester = ReconstructionTester()
            return tester.test_synthetic_data() and tester.test_full_pipeline()
        
        else:
            print(f"❌ Módulo desconocido: {module_name}")
            print("Módulos disponibles: mmpose, calibration, triangulation, reconstruction")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando tests del módulo {module_name}: {e}")
        return False

def print_final_summary(results: Dict[str, bool], total_time: float):
    """Imprimir resumen final de todos los tests"""
    print("\n" + "🦴" + "=" * 68 + "🦴")
    print("📊 RESUMEN FINAL DE TESTS")
    print("🦴" + "=" * 68 + "🦴")
    
    # Estadísticas
    total_modules = len(results)
    passed_modules = sum(1 for success in results.values() if success)
    failed_modules = total_modules - passed_modules
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   • Total módulos testados: {total_modules}")
    print(f"   • Módulos exitosos: {passed_modules}")
    print(f"   • Módulos fallidos: {failed_modules}")
    print(f"   • Tiempo total: {total_time:.1f} segundos")
    print(f"   • Tasa de éxito: {(passed_modules/total_modules)*100:.1f}%")
    
    print(f"\n📋 DETALLE POR MÓDULO:")
    for module, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   • {module.capitalize():15} {status}")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    if failed_modules == 0:
        print("   🎉 ¡Todos los tests pasaron! Sistema listo para producción.")
        print("   🚀 Puedes proceder con:")
        print("      - python download_models.py  # Descargar modelos MMPose")
        print("      - python app.py              # Iniciar servidor")
    else:
        print("   ⚠️  Algunos tests fallaron. Revisa los errores arriba.")
        print("   🔧 Acciones sugeridas:")
        
        if not results.get('mmpose', True):
            print("      - Instalar MMPose: pip install mmpose")
            print("      - Verificar GPU: nvidia-smi")
        
        if not results.get('calibration', True):
            print("      - Instalar OpenCV: pip install opencv-python")
            print("      - Verificar numpy: pip install numpy")
        
        if not results.get('triangulation', True):
            print("      - Verificar cálculos numéricos")
            print("      - Revisar configuración de cámaras")
        
        if not results.get('reconstruction', True):
            print("      - Verificar pipeline completo")
            print("      - Revisar imports de módulos")
    
    # Estado del sistema
    print(f"\n🎯 ESTADO DEL SISTEMA:")
    if failed_modules == 0:
        print("   🟢 SISTEMA OPERATIVO - Listo para usar")
    elif failed_modules <= 1:
        print("   🟡 SISTEMA FUNCIONAL - Con limitaciones menores")
    else:
        print("   🔴 SISTEMA REQUIERE ATENCIÓN - Múltiples fallos")
    
    print("\n🦴" + "=" * 68 + "🦴")

def main():
    """Función principal del test runner"""
    parser = argparse.ArgumentParser(
        description="Test Runner para Sistema de Análisis de Marcha",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python -m backend.tests.run_all_tests                    # Suite completa
  python -m backend.tests.run_all_tests --quick            # Tests rápidos
  python -m backend.tests.run_all_tests --module mmpose    # Solo MMPose
  python -m backend.tests.run_all_tests --module calibration  # Solo calibración
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Ejecutar solo tests rápidos sin dependencias pesadas'
    )
    
    parser.add_argument(
        '--module',
        choices=['mmpose', 'calibration', 'triangulation', 'reconstruction'],
        help='Ejecutar tests de un módulo específico'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Output detallado de tests'
    )
    
    args = parser.parse_args()
    
    # Configurar verbosidad
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Medir tiempo total
    start_time = time.time()
    
    try:
        # Ejecutar tests según argumentos
        if args.module:
            results = {args.module: run_module_tests(args.module)}
        elif args.quick:
            results = run_quick_tests()
        else:
            results = run_full_tests()
        
        # Calcular tiempo
        total_time = time.time() - start_time
        
        # Mostrar resumen
        print_final_summary(results, total_time)
        
        # Exit code basado en resultados
        all_passed = all(results.values())
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrumpidos por el usuario")
        return 130
    
    except Exception as e:
        print(f"\n💥 Error inesperado ejecutando tests: {e}")
        logger.exception("Error en test runner")
        return 1

if __name__ == "__main__":
    sys.exit(main())

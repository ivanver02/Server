#!/usr/bin/env python3
"""
Script para verificar qué datos de keypoints 2D están disponibles
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio del proyecto al path
sys.path.append(str(Path(__file__).parent.parent.parent))

def check_available_data(base_data_dir: Path):
    """Verificar qué datos están disponibles"""
    keypoints_dir = base_data_dir / "processed" / "2D_keypoints"
    
    if not keypoints_dir.exists():
        print(f"❌ Directorio no encontrado: {keypoints_dir}")
        return
    
    print(f"📁 Explorando datos en: {keypoints_dir}")
    print("=" * 60)
    
    # Buscar pacientes
    patients = []
    for patient_dir in keypoints_dir.iterdir():
        if patient_dir.is_dir() and patient_dir.name.startswith('patient'):
            patients.append(patient_dir)
    
    if not patients:
        print("❌ No se encontraron directorios de pacientes")
        return
    
    for patient_dir in sorted(patients):
        patient_id = patient_dir.name.replace('patient', '')
        print(f"\n👤 Paciente {patient_id}:")
        
        # Buscar sesiones
        sessions = []
        for session_dir in patient_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session'):
                sessions.append(session_dir)
        
        if not sessions:
            print("   ❌ No se encontraron sesiones")
            continue
        
        for session_dir in sorted(sessions):
            session_id = session_dir.name.replace('session', '')
            print(f"   📋 Sesión {session_id}:")
            
            # Buscar cámaras
            cameras = []
            for camera_dir in session_dir.iterdir():
                if camera_dir.is_dir() and camera_dir.name.startswith('camera'):
                    cameras.append(camera_dir)
            
            if not cameras:
                print("      ❌ No se encontraron cámaras")
                continue
            
            for camera_dir in sorted(cameras):
                camera_id = camera_dir.name.replace('camera', '')
                coord_dir = camera_dir / "coordinates"
                conf_dir = camera_dir / "confidence"
                
                coord_files = list(coord_dir.glob("*.npy")) if coord_dir.exists() else []
                conf_files = list(conf_dir.glob("*.npy")) if conf_dir.exists() else []
                
                if coord_files and conf_files:
                    print(f"      📸 Cámara {camera_id}: {len(coord_files)} archivos")
                    
                    # Verificar un archivo como ejemplo
                    if coord_files:
                        sample_file = coord_files[0]
                        try:
                            data = np.load(sample_file)
                            print(f"         📊 Ejemplo {sample_file.name}: shape={data.shape}, dtype={data.dtype}")
                        except Exception as e:
                            print(f"         ❌ Error leyendo {sample_file.name}: {e}")
                else:
                    print(f"      ❌ Cámara {camera_id}: datos incompletos")

def check_specific_session(base_data_dir: Path, patient_id: str, session_id: str):
    """Verificar datos específicos de una sesión"""
    session_dir = base_data_dir / "processed" / "2D_keypoints" / f"patient{patient_id}" / f"session{session_id}"
    
    print(f"\n🔍 Análisis detallado: Paciente {patient_id}, Sesión {session_id}")
    print("=" * 60)
    
    if not session_dir.exists():
        print(f"❌ Sesión no encontrada: {session_dir}")
        return
    
    camera_dirs = [d for d in session_dir.iterdir() if d.is_dir() and d.name.startswith('camera')]
    
    if not camera_dirs:
        print("❌ No se encontraron cámaras")
        return
    
    print(f"📸 Cámaras encontradas: {len(camera_dirs)}")
    
    for camera_dir in sorted(camera_dirs):
        camera_id = camera_dir.name.replace('camera', '')
        print(f"\n📷 Cámara {camera_id}:")
        
        coord_dir = camera_dir / "coordinates"
        conf_dir = camera_dir / "confidence"
        
        if not coord_dir.exists() or not conf_dir.exists():
            print("   ❌ Directorios de coordenadas o confianza faltantes")
            continue
        
        coord_files = sorted(coord_dir.glob("*.npy"))
        conf_files = sorted(conf_dir.glob("*.npy"))
        
        print(f"   📊 Archivos de coordenadas: {len(coord_files)}")
        print(f"   📊 Archivos de confianza: {len(conf_files)}")
        
        # Verificar correspondencia
        coord_stems = {f.stem for f in coord_files}
        conf_stems = {f.stem for f in conf_files}
        
        matching = coord_stems.intersection(conf_stems)
        print(f"   ✅ Archivos correspondientes: {len(matching)}")
        
        if matching:
            # Analizar un archivo de ejemplo
            sample_stem = next(iter(matching))
            coord_file = coord_dir / f"{sample_stem}.npy"
            conf_file = conf_dir / f"{sample_stem}.npy"
            
            try:
                coords = np.load(coord_file)
                confidences = np.load(conf_file)
                
                print(f"   📈 Ejemplo {sample_stem}:")
                print(f"      - Coordenadas: shape={coords.shape}, rango=({coords.min():.1f}, {coords.max():.1f})")
                print(f"      - Confianzas: shape={confidences.shape}, rango=({confidences.min():.3f}, {confidences.max():.3f})")
                
                # Estadísticas de confianza
                high_conf = np.sum(confidences > 0.5)
                medium_conf = np.sum((confidences > 0.3) & (confidences <= 0.5))
                low_conf = np.sum(confidences <= 0.3)
                
                print(f"      - Confianza alta (>0.5): {high_conf}/{len(confidences)} keypoints")
                print(f"      - Confianza media (0.3-0.5): {medium_conf}/{len(confidences)} keypoints")
                print(f"      - Confianza baja (≤0.3): {low_conf}/{len(confidences)} keypoints")
                
            except Exception as e:
                print(f"   ❌ Error leyendo archivos: {e}")

def main():
    """Ejecutar verificación de datos"""
    base_data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("🔍 Verificación de Datos de Keypoints 2D")
    print("=" * 60)
    
    # Exploración general
    check_available_data(base_data_dir)
    
    # Análisis específico del paciente 1, sesión 8
    check_specific_session(base_data_dir, "1", "8")
    
    print("\n" + "=" * 60)
    print("✅ Verificación completada")

if __name__ == "__main__":
    main()

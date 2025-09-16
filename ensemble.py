from backend.processing.ensemble import EnsembleProcessor
from config import data_config

if __name__ == "__main__":
    # Procesar ensemble para paciente 57, sesión 57, chunk 7
    ensemble_processor = EnsembleProcessor(data_config.base_data_dir)
    ensemble_processor.process_session_ensemble(57, 57, 7)
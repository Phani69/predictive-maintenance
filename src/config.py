import os

class Config:
    WINDOW_SIZE = 1000  # Samples per window (1 sec at 1kHz)
    VALID_EXTENSIONS = ('.txt', '.csv', '')  # Include files without extensions
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SMOTE_RATIO = 1.0
    DATA_DIR = "data"  # Directory with .rar files
    EXTRACT_DIR = "temp_extracted"  # Directory for extracted files
    MODEL_DIR = "models"
    PLOT_DIR = "plots"
    LOG_FILE = "maintenance_model.log"
    BATCH_SIZE = 100  # Process 100 files at a time to reduce memory usage
    SAMPLE_RATE = 0.5  # Sample 10% of data points to reduce computation
    
    @staticmethod
    def ensure_dirs():
        """Ensure output directories exist."""
        for directory in [Config.DATA_DIR, Config.EXTRACT_DIR, Config.MODEL_DIR, Config.PLOT_DIR]:
            os.makedirs(directory, exist_ok=True)
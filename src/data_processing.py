import pandas as pd
import os
import logging
from config import Config
import glob
import numpy as np

class DataProcessor:
    """Handles loading and preprocessing of NASA Bearing Dataset from extracted subdirectories."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """Load and preprocess data from extracted subdirectories."""
        self.logger.info("Loading NASA Bearing Dataset from temp_extracted...")
        
        Config.ensure_dirs()
        
        # Find subdirectories in temp_extracted
        subfolders = [f for f in glob.glob(os.path.join(Config.EXTRACT_DIR, "*")) if os.path.isdir(f)]
        if not subfolders:
            self.logger.error("No subdirectories found in temp_extracted")
            raise ValueError("No subdirectories found")
        
        all_data = []
        file_count = 0
        for subfolder in subfolders:
            files = glob.glob(os.path.join(subfolder, "*"))
            total_files = len(files)
            failure_threshold = int(0.7 * total_files)  # Last 30% faulty
            self.logger.info(f"Processing subdirectory {subfolder} with {total_files} files")
            
            # Process files in batches
            for batch_start in range(0, total_files, Config.BATCH_SIZE):
                batch_files = files[batch_start:batch_start + Config.BATCH_SIZE]
                for idx, file in enumerate(batch_files):
                    try:
                        # Try multiple separators for extension-less files
                        for sep in [' ', '\t', ',']:
                            try:
                                data = pd.read_csv(file, sep=sep, header=None)
                                if data.shape[1] >= 2:
                                    break
                            except:
                                continue
                        else:
                            self.logger.warning(f"Could not parse {file}")
                            continue
                        
                        # Sample data to reduce memory usage
                        if Config.SAMPLE_RATE < 1.0:
                            data = data.sample(frac=Config.SAMPLE_RATE, random_state=Config.RANDOM_STATE)
                        
                        data['filename'] = os.path.basename(file)
                        data['file_index'] = file_count
                        data['subfolder'] = os.path.basename(subfolder)
                        data['label'] = 1 if (batch_start + idx) >= failure_threshold else 0
                        all_data.append(data)
                        file_count += 1
                        if file_count % 50 == 0:
                            self.logger.info(f"Loaded {file_count} files...")
                    except Exception as e:
                        self.logger.warning(f"Error loading {file}: {e}")
                        continue
        
        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Dataset loaded: {self.raw_data.shape}")
            self.raw_data = self.raw_data.dropna()
            self.processed_data = self.raw_data.copy()
            self.logger.info(f"After removing missing data: {self.raw_data.shape}")
            self.logger.info(f"Class distribution:\n{self.processed_data['label'].value_counts()}")
        else:
            self.logger.error("No data loaded")
            raise ValueError("No data loaded")
        
        return self.processed_data
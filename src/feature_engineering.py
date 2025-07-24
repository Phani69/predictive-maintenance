import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from tqdm import tqdm
import pandas as pd
from config import Config
import logging

class FeatureExtractor:
    """Extracts time and frequency domain features from sensor data."""
    
    def __init__(self):
        self.features_df = None
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, processed_data):
        """Extract features from sensor data."""
        self.logger.info("Extracting features...")
        
        if processed_data is None or processed_data.empty:
            self.logger.error("No processed data available")
            raise ValueError("No processed data available")
        
        # Dynamically select sensor columns (numeric columns excluding metadata)
        sensor_columns = [col for col in processed_data.columns if isinstance(col, int)]
        if not sensor_columns:
            self.logger.error("No sensor columns found")
            raise ValueError("No sensor columns found")
        
        feature_list = []
        window_size = Config.WINDOW_SIZE
        
        # Process data in smaller groups by file_index
        for file_idx in tqdm(processed_data['file_index'].unique()):
            file_data = processed_data[processed_data['file_index'] == file_idx]
            for idx in range(0, len(file_data) - window_size, window_size):
                window_data = file_data.iloc[idx:idx + window_size]
                if len(window_data) < window_size:
                    continue
                
                window_features = {
                    'label': window_data['label'].mode()[0],
                    'file_index': file_idx,
                    'subfolder': window_data['subfolder'].iloc[0]
                }
                
                for col in sensor_columns:
                    signal = window_data[col].values
                    # Time-domain features (reduced set for efficiency)
                    window_features[f'ch{col}_mean'] = np.mean(signal)
                    window_features[f'ch{col}_rms'] = np.sqrt(np.mean(signal**2))
                    window_features[f'ch{col}_std'] = np.std(signal)
                    
                    # Reduced frequency-domain features
                    fft_vals = fft(signal)
                    freqs = fftfreq(len(signal), 1/1000)
                    magnitude = np.abs(fft_vals)
                    window_features[f'ch{col}_fft_low'] = magnitude[(freqs >= 0) & (freqs < 100)].sum()
                    mag_half = magnitude[:len(magnitude)//2]
                    freq_half = freqs[:len(freqs)//2]
                    sum_mag = np.sum(mag_half)
                    window_features[f'ch{col}_spectral_centroid'] = np.sum(freq_half * mag_half) / sum_mag if sum_mag != 0 else 0
                
                feature_list.append(window_features)
        
        self.features_df = pd.DataFrame(feature_list)
        self.logger.info(f"Feature extraction complete: {self.features_df.shape}")
        return self.features_df
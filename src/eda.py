import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import Config
import logging
import os

class EDA:
    """Performs exploratory data analysis with visualizations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def perform_eda(self, features_df):
        """Generate EDA visualizations."""
        self.logger.info("Performing EDA...")
        
        if features_df is None or features_df.empty:
            self.logger.error("No features available for EDA")
            raise ValueError("No features available for EDA")
        
        Config.ensure_dirs()
        feature_cols = [col for col in features_df.columns if col not in ['label', 'file_index', 'subfolder']]
        
        plt.figure(figsize=(20, 15))
        
        # Class distribution
        plt.subplot(3, 4, 1)
        class_counts = features_df['label'].value_counts()
        plt.bar(['Healthy', 'Faulty'], class_counts.values)
        plt.title('Class Distribution')
        plt.ylabel('Count')
        for i, v in enumerate(class_counts.values):
            plt.text(i, v + 5, str(v), ha='center')
        
        # RMS plot
        rms_col = next((col for col in feature_cols if 'rms' in col), feature_cols[0])
        healthy = features_df[features_df['label'] == 0]
        faulty = features_df[features_df['label'] == 1]
        plt.subplot(3, 4, 2)
        plt.plot(healthy['file_index'], healthy[rms_col], 'g.', alpha=0.6, label='Healthy')
        plt.plot(faulty['file_index'], faulty[rms_col], 'r.', alpha=0.6, label='Faulty')
        plt.title(f'{rms_col} Over Time')
        plt.xlabel('File Index')
        plt.ylabel(f'{rms_col} Value')
        plt.legend()
        
        # RMS histogram
        plt.subplot(3, 4, 3)
        plt.hist(healthy[rms_col], alpha=0.7, label='Healthy', bins=30)
        plt.hist(faulty[rms_col], alpha=0.7, label='Faulty', bins=30)
        plt.title(f'{rms_col} Distribution')
        plt.xlabel(f'{rms_col} Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        # RMS boxplot
        plt.subplot(3, 4, 4)
        features_df.boxplot(column=rms_col, by='label', ax=plt.gca())
        plt.title(f'{rms_col} Boxplot by Class')
        plt.suptitle('')
        
        # Correlation matrix
        plt.subplot(3, 4, 6)
        corr_matrix = features_df[feature_cols[:10]].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # PCA visualization
        plt.subplot(3, 4, 7)
        X_scaled = StandardScaler().fit_transform(features_df[feature_cols].fillna(0))
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        plt.scatter(X_pca[healthy.index, 0], X_pca[healthy.index, 1], c='green', alpha=0.6, label='Healthy')
        plt.scatter(X_pca[faulty.index, 0], X_pca[faulty.index, 1], c='red', alpha=0.6, label='Faulty')
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOT_DIR, 'eda_analysis.png'), dpi=300)
        plt.show()
        self.logger.info("EDA Complete")
import shap
import matplotlib.pyplot as plt
import numpy as np
from config import Config
import logging
import os

class Explainer:
    """Explains model predictions using SHAP."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def explain_model(self, model, X_test_scaled, feature_names):
        """Generate SHAP explanations."""
        self.logger.info("Generating SHAP explanations...")
        
        if model is None:
            self.logger.error("No trained model available")
            raise ValueError("No trained model available")
        
        Config.ensure_dirs()
        
        # Limit to available samples (max 50 for efficiency)
        sample_size = min(50, X_test_scaled.shape[0])
        X_test_subset = X_test_scaled[:sample_size]
        self.logger.info(f"Using {sample_size} samples for SHAP explanation")
        self.logger.info(f"X_test_subset shape: {X_test_subset.shape}")
        self.logger.info(f"Feature names: {feature_names}")
        
        try:
            if hasattr(model, 'feature_importances_'):
                self.logger.info("Using TreeExplainer for tree-based model")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_subset)
                self.logger.info(f"SHAP values shape: {np.array(shap_values).shape}")
                
                # For binary classification, shap_values is a list [neg_class, pos_class]
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values_pos = shap_values[1]  # Positive class
                else:
                    shap_values_pos = shap_values  # Single output (e.g., XGBoost with binary output)
                
                shap.summary_plot(shap_values_pos, X_test_subset, feature_names=feature_names, show=False)
            else:
                self.logger.info("Using KernelExplainer for non-tree model")
                explainer = shap.KernelExplainer(model.predict_proba, X_test_subset)
                shap_values = explainer.shap_values(X_test_subset)
                self.logger.info(f"SHAP values shape: {np.array(shap_values).shape}")
                
                # KernelExplainer returns list for binary classification
                shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
                
                shap.summary_plot(shap_values_pos, X_test_subset, feature_names=feature_names, show=False)
            
            plt.savefig(os.path.join(Config.PLOT_DIR, 'shap_analysis.png'), dpi=300)
            plt.close()  # Close plot to free memory
            self.logger.info("Explainability complete")
        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {str(e)}")
            raise ValueError(f"SHAP explanation failed: {str(e)}")
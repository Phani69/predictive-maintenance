from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from config import Config
import logging
import os
import numpy as np

class ModelTrainer:
    """Trains and evaluates predictive maintenance models."""
    
    def __init__(self):
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def train_models(self, features_df):
        """Train and evaluate multiple ML models."""
        self.logger.info("Training models...")
        
        if features_df is None or features_df.empty:
            self.logger.error("No features available")
            raise ValueError("No features available")
        
        feature_cols = [col for col in features_df.columns if col not in ['label', 'file_index', 'subfolder']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['label']
        
        # Split by file_index to avoid temporal leakage
        unique_files = features_df['file_index'].unique()
        train_files, test_files = train_test_split(unique_files, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
        train_idx = features_df['file_index'].isin(train_files)
        self.X_train, self.X_test = X[train_idx], X[~train_idx]
        self.y_train, self.y_test = y[train_idx], y[~train_idx]
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Apply SMOTE
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train_scaled, self.y_train)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=Config.RANDOM_STATE),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE),
            'XGBoost': xgb.XGBClassifier(random_state=Config.RANDOM_STATE, eval_metric='logloss')
        }
        
        results = {}
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            model.fit(X_train_balanced, y_train_balanced)
            y_pred = model.predict(self.X_test_scaled)
            results[name] = {
                'model': model,
                'f1_score': f1_score(self.y_test, y_pred),
                'y_pred': y_pred
            }
            self.logger.info(f"{name} F1 Score: {results[name]['f1_score']:.3f}")
            print(classification_report(self.y_test, y_pred))
        
        self.best_model = results[max(results, key=lambda k: results[k]['f1_score'])]['model']
        self.models = results
        self.logger.info(f"Best model: {max(results, key=lambda k: results[k]['f1_score'])}")
        
        return results
    
    def validate_and_tune(self):
        """Validate and tune the best model."""
        self.logger.info("Validating and tuning models...")
        
        if self.best_model is None:
            self.logger.error("No trained model available")
            raise ValueError("No trained model available")
        
        Config.ensure_dirs()
        
        # Cross-validation
        cv_scores = cross_val_score(self.best_model, self.X_train_scaled, self.y_train, cv=5, scoring='f1')
        self.logger.info(f"CV F1 Mean: {cv_scores.mean():.3f}")
        
        # Hyperparameter tuning
        if isinstance(self.best_model, RandomForestClassifier):
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
            grid_search = GridSearchCV(self.best_model, param_grid, cv=3, scoring='f1')
            grid_search.fit(self.X_train_scaled, self.y_train)
            self.best_model = grid_search.best_estimator_
            self.logger.info(f"Best Random Forest params: {grid_search.best_params_}")
        elif isinstance(self.best_model, xgb.XGBClassifier):
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9]}
            grid_search = GridSearchCV(self.best_model, param_grid, cv=3, scoring='f1')
            grid_search.fit(self.X_train_scaled, self.y_train)
            self.best_model = grid_search.best_estimator_
            self.logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        
        # Noise robustness test
        X_test_noisy = self.X_test_scaled + np.random.normal(0, 0.1, self.X_test_scaled.shape)
        f1_noisy = f1_score(self.y_test, self.best_model.predict(X_test_noisy))
        self.logger.info(f"F1 with noise: {f1_noisy:.3f}")
        
        # Confusion matrix
        plt.figure(figsize=(10, 5))
        cm = confusion_matrix(self.y_test, self.models[list(self.models.keys())[0]]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(Config.PLOT_DIR, 'validation_results.png'), dpi=300)
        plt.show()
        self.logger.info("Validation complete")
    
    def save_model_and_scaler(self):
        """Save the trained model and scaler."""
        self.logger.info("Saving model and scaler...")
        
        if self.best_model is None:
            self.logger.error("No trained model to save")
            raise ValueError("No trained model to save")
        
        Config.ensure_dirs()
        model_path = os.path.join(Config.MODEL_DIR, 'trained_model.pkl')
        scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.pkl')
        feature_cols_path = os.path.join(Config.MODEL_DIR, 'feature_columns.pkl')
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        feature_cols = [col for col in self.X_train.columns]
        with open(feature_cols_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Scaler saved to {scaler_path}")
        self.logger.info(f"Feature columns saved to {feature_cols_path}")
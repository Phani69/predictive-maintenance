import logging
from data_processing import DataProcessor
from feature_engineering import FeatureExtractor
from eda import EDA
from model_training import ModelTrainer
from explainability import Explainer
from config import Config

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=Config.LOG_FILE
    )

def run_pipeline(data_path):
    """Run the full predictive maintenance pipeline."""
    print("\n" + "="*50)
    print("RUNNING FULL TRAINING PIPELINE")
    print("="*50)
    
    try:
        # Initialize components
        processor = DataProcessor(data_path)
        extractor = FeatureExtractor()
        eda = EDA()
        trainer = ModelTrainer()
        explainer = Explainer()
        
        # Run pipeline
        processed_data = processor.load_data()
        logging.info("Starting feature extraction")
        features_df = extractor.extract_features(processed_data)
        eda.perform_eda(features_df)
        trainer.train_models(features_df)
        trainer.validate_and_tune()
        
        # Pass correct feature columns (excluding metadata)
        feature_cols = [col for col in features_df.columns if col not in ['label', 'file_index', 'subfolder']]
        explainer.explain_model(trainer.best_model, trainer.X_test_scaled, feature_cols)
        trainer.save_model_and_scaler()
        
        print("✅ Training pipeline completed successfully")
        print("✅ Model and scaler saved for GUI usage")
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    data_path = "/Users/phanikoduri/predictive_maintenance_project/data"
    run_pipeline(data_path)
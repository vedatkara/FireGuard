import joblib
import numpy as np
from pathlib import Path

def load_model():
    """Load the trained model"""
    try:
        model_path = Path('Joblib/best_model.joblib')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        print(f"Model type: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Model features: {model.feature_names_in_}")
            print(f"Number of model features: {len(model.feature_names_in_)}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def load_feature_names():
    """Load the feature names"""
    try:
        feature_names_path = Path('Joblib/feature_names.joblib')
        if not feature_names_path.exists():
            raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")
        
        feature_names = joblib.load(feature_names_path)
        print(f"Raw feature names: {feature_names}")
        
        # Ensure feature_names is a list
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        
        print(f"Final feature names: {feature_names}")
        print(f"Number of features: {len(feature_names)}")
        
        return feature_names
    except Exception as e:
        print(f"Error loading feature names: {str(e)}")
        raise 
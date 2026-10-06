import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """Application Configuration Settings"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cnc-pred-maint-secret-key-2026-mca')
    
    # Database Settings (XAMPP / MySQL compatible)
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DB = os.environ.get('MYSQL_DB', 'cnc_maintenance_db')
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
    
    # Path settings
    WEB_APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(WEB_APP_DIR)
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    
    # Preprocessing Artifacts
    IMPUTER_PATH = os.path.join(MODELS_DIR, 'imputer.pkl')
    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    # Machine Learning Model Paths
    MODELS_CONFIG = {
        'xgboost': {
            'name': 'XGBoost (SMOTE Champion)',
            'path': os.path.join(MODELS_DIR, 'best_model', 'champion_model.pkl'),
            'fallback_path': os.path.join(MODELS_DIR, 'xgboost', 'xgboost_smote_model.pkl'),
            'requires_scaling': False,
            'description': 'Extreme Gradient Boosted Trees with SMOTE balancing (Highest Recall 89.22%, Ultra-low latency)'
        },
        'random_forest': {
            'name': 'Random Forest Classifier',
            'path': os.path.join(MODELS_DIR, 'random_forest', 'random_forest_smote_model.pkl'),
            'fallback_path': os.path.join(MODELS_DIR, 'random_forest', 'random_forest_model.pkl'),
            'requires_scaling': False,
            'description': 'Bagging Ensemble of 100 Decision Trees (Robust against sensor outliers, 85.03% Recall)'
        },
        'svm': {
            'name': 'Support Vector Machine (RBF)',
            'path': os.path.join(MODELS_DIR, 'svm', 'svm_smote_model.pkl'),
            'fallback_path': os.path.join(MODELS_DIR, 'svm', 'svm_model.pkl'),
            'requires_scaling': True,
            'description': 'Non-linear Maximum Margin Classifier with RBF Kernel (Uses Standardized features, 89.22% Recall)'
        }
    }
    
    # Active default model (configured from ML comparison champion)
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'xgboost')
    
    # Risk Classification Thresholds
    RISK_THRESHOLDS = {
        'low': 0.35,      # 0% - 34.99% -> Low Risk (Green)
        'medium': 0.65    # 35% - 64.99% -> Medium Risk (Amber), >= 65% -> High Risk (Red)
    }
    
    # Exactly 5 designated sensor features
    SENSOR_FEATURES = [
        'vibration_rms',
        'temperature_motor',
        'current_phase_avg',
        'pressure_level',
        'rpm'
    ]

import os
import joblib
import pandas as pd
import numpy as np
from config import Config

class ModelLoader:
    """
    Centralized Machine Learning Model & Preprocessor Loader.
    Provides leak-free inference pipeline using artifacts saved from training notebooks.
    """
    
    _imputer = None
    _scaler = None
    _loaded_models = {}

    @classmethod
    def load_preprocessors(cls):
        """Load and cache the median imputer and standard scaler."""
        if cls._imputer is None:
            if not os.path.exists(Config.IMPUTER_PATH):
                raise FileNotFoundError(f"Imputer artifact missing at: {Config.IMPUTER_PATH}")
            cls._imputer = joblib.load(Config.IMPUTER_PATH)

        if cls._scaler is None:
            if not os.path.exists(Config.SCALER_PATH):
                raise FileNotFoundError(f"Scaler artifact missing at: {Config.SCALER_PATH}")
            cls._scaler = joblib.load(Config.SCALER_PATH)

        return cls._imputer, cls._scaler

    @classmethod
    def get_model(cls, model_key=None):
        """
        Dynamically retrieve and cache the requested model.
        Available keys: 'xgboost', 'random_forest', 'svm'
        """
        key = (model_key or Config.DEFAULT_MODEL).lower()
        if key not in Config.MODELS_CONFIG:
            key = Config.DEFAULT_MODEL

        if key not in cls._loaded_models:
            model_info = Config.MODELS_CONFIG[key]
            model_path = model_info['path']
            
            # Use fallback path if primary does not exist
            if not os.path.exists(model_path):
                model_path = model_info.get('fallback_path', model_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Trained model file not found for '{key}' at: {model_path}")

            cls._loaded_models[key] = joblib.load(model_path)
            print(f"[ModelLoader] Loaded '{key}' model from: {model_path}")

        return cls._loaded_models[key], Config.MODELS_CONFIG[key]

    @classmethod
    def predict(cls, vibration_rms, temperature_motor, current_phase_avg, pressure_level, rpm, model_key=None):
        """
        Execute end-to-end inference pipeline:
        1. Form input DataFrame with the 5 sensor features
        2. Impute missing values with trained SimpleImputer
        3. If SVM: Scale features with trained StandardScaler
        4. Predict class (0 or 1) and exact failure probability
        5. Map to risk level and maintenance recommendation
        """
        imputer, scaler = cls.load_preprocessors()
        model, model_info = cls.get_model(model_key)

        # 1. Input DataFrame
        input_data = pd.DataFrame([{
            'vibration_rms': float(vibration_rms),
            'temperature_motor': float(temperature_motor),
            'current_phase_avg': float(current_phase_avg),
            'pressure_level': float(pressure_level),
            'rpm': float(rpm)
        }], columns=Config.SENSOR_FEATURES)

        # 2. Imputation
        imputed_array = imputer.transform(input_data)
        features_for_model = pd.DataFrame(imputed_array, columns=Config.SENSOR_FEATURES)

        # 3. Scaling (Strictly if model requires it, e.g. SVM)
        if model_info.get('requires_scaling', False):
            scaled_array = scaler.transform(features_for_model)
            features_for_model = pd.DataFrame(scaled_array, columns=Config.SENSOR_FEATURES)

        # 4. Model Prediction & Probability
        pred_class = int(model.predict(features_for_model)[0])
        prob_failure = float(model.predict_proba(features_for_model)[0, 1])
        prob_pct = round(prob_failure * 100, 2)

        # 5. Risk Level Interpretation
        if prob_failure < Config.RISK_THRESHOLDS['low']:
            risk_level = "Low Risk"
            risk_badge = "success"
        elif prob_failure < Config.RISK_THRESHOLDS['medium']:
            risk_level = "Medium Risk"
            risk_badge = "warning"
        else:
            risk_level = "High Risk"
            risk_badge = "danger"
        return {
            'prediction': pred_class,
            'prediction_label': 'FAILURE LIKELY' if pred_class == 1 else 'NORMAL OPERATION',
            'failure_probability': prob_pct,
            'probability_ratio': prob_failure,
            'risk_level': risk_level,
            'risk_badge': risk_badge,
            'recommendation': '',
            'model_key': model_key or Config.DEFAULT_MODEL,
            'model_name': model_info['name'],
            'inputs': {
                'vibration_rms': float(vibration_rms),
                'temperature_motor': float(temperature_motor),
                'current_phase_avg': float(current_phase_avg),
                'pressure_level': float(pressure_level),
                'rpm': float(rpm)
            }
        }

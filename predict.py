"""
CNC Predictive Maintenance — Real-Time Inference Script
Champion Model: XGBoost Classifier (Trained with SMOTE)
Target: failure_within_24h (0 = Normal, 1 = Impending Failure)
"""

import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_model")
MODEL_PATH = os.path.join(MODEL_DIR, "champion_model.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm"
]

def load_inference_pipeline():
    """Load champion model and imputer."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(IMPUTER_PATH):
        raise FileNotFoundError(f"Imputer file not found at: {IMPUTER_PATH}")
    
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, imputer

def predict_cnc_status(vibration_rms, temperature_motor, current_phase_avg, pressure_level, rpm):
    """
    Perform single-sample CNC failure prediction.
    
    Parameters:
        vibration_rms (float): Spindle vibration RMS in mm/s
        temperature_motor (float): Motor temperature in °C
        current_phase_avg (float): Phase current draw in Amperes
        pressure_level (float): Fluid/clamping pressure in bar
        rpm (float): Spindle rotational speed in RPM
        
    Returns:
        dict: Prediction result with failure probability and status.
    """
    model, imputer = load_inference_pipeline()
    
    # Construct DataFrame
    input_df = pd.DataFrame([{
        "vibration_rms": vibration_rms,
        "temperature_motor": temperature_motor,
        "current_phase_avg": current_phase_avg,
        "pressure_level": pressure_level,
        "rpm": rpm
    }], columns=FEATURES)
    
    # Impute missing values if any
    imputed_values = imputer.transform(input_df)
    imputed_df = pd.DataFrame(imputed_values, columns=FEATURES)
    
    # Predict
    prob_failure = float(model.predict_proba(imputed_df)[0, 1])
    pred_class = int(model.predict(imputed_df)[0])
    
    status = "FAILURE IMMINENT (Within 24 Hours)" if pred_class == 1 else "NORMAL (Machine Healthy)"
    risk_level = "CRITICAL" if prob_failure >= 0.70 else ("WARNING" if prob_failure >= 0.40 else "SAFE")
    
    return {
        "status": status,
        "prediction_code": pred_class,
        "failure_probability_pct": round(prob_failure * 100, 2),
        "risk_level": risk_level,
        "inputs": {
            "vibration_rms": vibration_rms,
            "temperature_motor": temperature_motor,
            "current_phase_avg": current_phase_avg,
            "pressure_level": pressure_level,
            "rpm": rpm
        }
    }

def main():
    parser = argparse.ArgumentParser(description="CNC Machine Predictive Maintenance Real-Time Inference")
    parser.add_argument("--vibration", type=float, default=None, help="Vibration RMS (mm/s)")
    parser.add_argument("--temperature", type=float, default=None, help="Motor Temperature (°C)")
    parser.add_argument("--current", type=float, default=None, help="Current Phase Avg (A)")
    parser.add_argument("--pressure", type=float, default=None, help="Pressure Level (bar)")
    parser.add_argument("--rpm", type=float, default=None, help="Rotational Speed (RPM)")
    parser.add_argument("--file", type=str, default=None, help="Path to CSV file with sensor readings for batch inference")
    parser.add_argument("--output", type=str, default=None, help="Path to save batch predictions CSV")

    args = parser.parse_args()

    if args.file:
        model, imputer = load_inference_pipeline()
        df = pd.read_csv(args.file)
        missing_cols = [c for c in FEATURES if c not in df.columns]
        if missing_cols:
            print(f"Error: CSV is missing required columns: {missing_cols}")
            sys.exit(1)
        
        imputed_data = imputer.transform(df[FEATURES])
        df_imputed = pd.DataFrame(imputed_data, columns=FEATURES)
        
        df["failure_prediction"] = model.predict(df_imputed)
        df["failure_probability"] = model.predict_proba(df_imputed)[:, 1]
        df["status"] = np.where(df["failure_prediction"] == 1, "FAILURE IMMINENT", "NORMAL")
        
        out_path = args.output or "batch_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"Batch predictions saved to: {out_path}")
        print(f"Total Rows Processed: {len(df):,}")
        print(f"Predicted Failures: {sum(df['failure_prediction'] == 1):,}")
        return

    # Check if CLI sensor inputs provided
    if any(v is not None for v in [args.vibration, args.temperature, args.current, args.pressure, args.rpm]):
        v = args.vibration if args.vibration is not None else 1.5
        t = args.temperature if args.temperature is not None else 45.0
        c = args.current if args.current is not None else 10.0
        p = args.pressure if args.pressure is not None else 50.0
        r = args.rpm if args.rpm is not None else 1200.0
        
        res = predict_cnc_status(v, t, c, p, r)
        print("\n" + "=" * 60)
        print("          CNC REAL-TIME PREDICTION RESULT")
        print("=" * 60)
        print(f"Machine Status      : {res['status']}")
        print(f"Risk Level          : {res['risk_level']}")
        print(f"Failure Probability : {res['failure_probability_pct']}%")
        print(f"Inputs              : {res['inputs']}")
        print("=" * 60 + "\n")
        return

    # Default interactive demonstration
    print("=" * 65)
    print("      CNC PREDICTIVE MAINTENANCE — CHAMPION MODEL (XGBOOST)")
    print("=" * 65)
    
    # Test Normal Sample
    normal_res = predict_cnc_status(vibration_rms=1.2, temperature_motor=42.0, current_phase_avg=9.0, pressure_level=55.0, rpm=1150.0)
    print(f"\n[Test 1: Healthy CNC Machine]")
    print(f"  Inputs: Vibration=1.2 mm/s, Temp=42.0 °C, Current=9.0 A, Pressure=55 bar, RPM=1150")
    print(f"  Status: {normal_res['status']}")
    print(f"  Risk Level: {normal_res['risk_level']} | Failure Probability: {normal_res['failure_probability_pct']}%")

    # Test Failure Sample
    failure_res = predict_cnc_status(vibration_rms=4.9, temperature_motor=84.0, current_phase_avg=25.0, pressure_level=30.0, rpm=1900.0)
    print(f"\n[Test 2: Degraded CNC Machine (Impending Breakdown)]")
    print(f"  Inputs: Vibration=4.9 mm/s, Temp=84.0 °C, Current=25.0 A, Pressure=30 bar, RPM=1900")
    print(f"  Status: {failure_res['status']}")
    print(f"  Risk Level: {failure_res['risk_level']} | Failure Probability: {failure_res['failure_probability_pct']}%")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

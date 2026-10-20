import os
import json
import numpy as np
import pandas as pd
from PyALE import ale
from config import Config
from models.model_loader import ModelLoader

class ALEService:
    """
    Accumulated Local Effects (ALE) Explainable AI Service for CNC Machine Telemetry.
    Computes and caches true main effect curves that avoid correlation bias on physical IoT sensor data.
    """
    
    _ale_cache = None
    CACHE_FILE = os.path.join(Config.MODELS_DIR, 'ale_curves.json')

    FEATURE_CONFIG = {
        'vibration_rms': {
            'name': 'Vibration RMS',
            'unit': 'mm/s',
            'icon': 'bi-activity',
            'desc': 'Spindle/bearing mechanical vibration level'
        },
        'temperature_motor': {
            'name': 'Motor Temperature',
            'unit': '°C',
            'icon': 'bi-thermometer-high',
            'desc': 'Core stator and winding thermal load'
        },
        'current_phase_avg': {
            'name': 'Phase Current',
            'unit': 'A',
            'icon': 'bi-lightning-charge',
            'desc': 'Electrical motor draw under spindle cutting load'
        },
        'pressure_level': {
            'name': 'Pressure Level',
            'unit': 'bar',
            'icon': 'bi-speedometer2',
            'desc': 'Hydraulic chuck and coolant line pressure'
        },
        'rpm': {
            'name': 'Spindle Speed (RPM)',
            'unit': 'RPM',
            'icon': 'bi-arrow-repeat',
            'desc': 'Rotational spindle cutting velocity'
        }
    }

    @classmethod
    def get_ale_curves(cls, force_recompute=False):
        """
        Loads cached ALE curves or precomputes them from X_train using PyALE.
        """
        if cls._ale_cache is not None and not force_recompute:
            return cls._ale_cache

        # Check file cache
        if os.path.exists(cls.CACHE_FILE) and not force_recompute:
            try:
                with open(cls.CACHE_FILE, 'r') as f:
                    cls._ale_cache = json.load(f)
                    return cls._ale_cache
            except Exception as e:
                print(f"[ALEService] Cache read warning: {e}. Recomputing curves...")

        # Compute curves using PyALE
        model, _ = ModelLoader.get_model()
        x_train_path = os.path.join(Config.PROJECT_DIR, 'dataset', 'processed', 'X_train.csv')
        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"Training dataset not found at: {x_train_path}")

        X_train = pd.read_csv(x_train_path)

        class ModelProbWrapper:
            def __init__(self, m):
                self.m = m
            def predict(self, X):
                return self.m.predict_proba(X)[:, 1]

        wrapped_model = ModelProbWrapper(model)
        curves = {}

        for feat in Config.SENSOR_FEATURES:
            res = ale(
                X=X_train,
                model=wrapped_model,
                feature=[feat],
                grid_size=40,
                include_CI=False,
                plot=False
            )
            xs = [round(float(x), 2) for x in res.index.values]
            ys = [round(float(y), 4) for y in res['eff'].values]
            
            # Find zero-crossing threshold where ALE transitions from safe (<0) to risky (>0)
            threshold = None
            for i in range(len(ys) - 1):
                if ys[i] <= 0 and ys[i+1] > 0:
                    threshold = round(xs[i], 2)
                    break
            if threshold is None:
                threshold = round(float(np.median(xs)), 2)

            curves[feat] = {
                'x': xs,
                'y': ys,
                'threshold': threshold,
                'min_val': float(np.min(xs)),
                'max_val': float(np.max(xs)),
                'min_eff': float(np.min(ys)),
                'max_eff': float(np.max(ys))
            }

        # Save to cache file
        try:
            with open(cls.CACHE_FILE, 'w') as f:
                json.dump(curves, f, indent=2)
        except Exception as e:
            print(f"[ALEService] Warning saving cache file: {e}")

        cls._ale_cache = curves
        return curves

    @classmethod
    def explain_prediction(cls, inputs, failure_probability, risk_level):
        """
        Generates comprehensive Explainable AI (ALE) explanation for a given prediction.
        
        Args:
            inputs: dict containing numeric sensor inputs:
                    {'vibration_rms': ..., 'temperature_motor': ..., 'current_phase_avg': ..., 'pressure_level': ..., 'rpm': ...}
            failure_probability: float percentage (e.g. 78.5)
            risk_level: 'High Risk', 'Medium Risk', or 'Low Risk'
            
        Returns:
            dict containing feature effects, summary text, and chart payload.
        """
        curves = cls.get_ale_curves()
        feature_effects = []
        positive_contributors = []
        negative_contributors = []

        chart_payload = {}

        for feat in Config.SENSOR_FEATURES:
            val = float(inputs.get(feat, 0))
            curve = curves.get(feat, {})
            xs = np.array(curve.get('x', []))
            ys = np.array(curve.get('y', []))
            threshold = curve.get('threshold', 0)

            # Interpolate ALE effect for the specific input value
            if len(xs) > 0 and len(ys) > 0:
                effect = float(np.interp(val, xs, ys))
            else:
                effect = 0.0

            conf = cls.FEATURE_CONFIG.get(feat, {'name': feat, 'unit': '', 'icon': 'bi-gear'})
            effect_pct = round(effect * 100, 1)

            # Determine risk contribution category
            if effect >= 0.08:
                status = "Critical Contributor"
                badge_class = "badge-critical"
                risk_color = "danger"
                icon_type = "bi-arrow-up-circle-fill text-danger"
            elif effect > 0.01:
                status = "Moderate Contributor"
                badge_class = "badge-atrisk"
                risk_color = "warning"
                icon_type = "bi-arrow-up-right-circle-fill text-warning"
            elif effect >= -0.05:
                status = "Neutral"
                badge_class = "badge-healthy"
                risk_color = "secondary"
                icon_type = "bi-dash-circle text-secondary"
            else:
                status = "Safe / Protective"
                badge_class = "badge-healthy"
                risk_color = "success"
                icon_type = "bi-arrow-down-circle-fill text-success"

            item = {
                'feature_key': feat,
                'name': conf['name'],
                'unit': conf['unit'],
                'icon': conf['icon'],
                'value': val,
                'effect': round(effect, 4),
                'effect_pct': effect_pct,
                'status': status,
                'badge_class': badge_class,
                'risk_color': risk_color,
                'icon_type': icon_type,
                'threshold': threshold,
                'is_above_threshold': val >= threshold if threshold else False
            }

            feature_effects.append(item)
            if effect > 0:
                positive_contributors.append(item)
            else:
                negative_contributors.append(item)

            # Chart payload data for interactive curve rendering
            chart_payload[feat] = {
                'name': conf['name'],
                'unit': conf['unit'],
                'x': curve.get('x', []),
                'y': curve.get('y', []),
                'current_x': val,
                'current_y': round(effect, 4),
                'threshold': threshold
            }

        # Sort feature effects by absolute impact
        feature_effects.sort(key=lambda x: abs(x['effect']), reverse=True)
        positive_contributors.sort(key=lambda x: x['effect'], reverse=True)

        # Generate human-readable narrative explanation
        if risk_level in ('High Risk', 'Medium Risk'):
            top_drivers = [f"{item['name']} ({item['value']} {item['unit']}, +{item['effect_pct']}%)" for item in positive_contributors[:2]]
            drivers_str = " and ".join(top_drivers) if top_drivers else "combined sensor telemetry"
            
            explanation_summary = (
                f"The model predicted {risk_level.lower()} ({failure_probability}%) primarily driven by {drivers_str}. "
                f"Accumulated Local Effects (ALE) shows these parameters pushed the machine beyond its normal empirical baseline."
            )
            
            # Actionable engineering recommendations
            recommendations = []
            for item in positive_contributors[:2]:
                if item['threshold'] and item['value'] > item['threshold']:
                    recommendations.append(
                        f"Reduce {item['name']} from {item['value']} {item['unit']} to below {item['threshold']} {item['unit']} to return into the safe ALE zone."
                    )
            if not recommendations:
                recommendations.append("Inspect mechanical spindle bearings, lubrication flow, and motor thermal ventilation.")
        else:
            explanation_summary = (
                f"All evaluated sensor readings are operating within safe empirical limits. "
                f"Accumulated Local Effects (ALE) curves show negative or nominal contributions across all parameters, confirming stable machine condition."
            )
            recommendations = [
                "Maintain standard operating speeds and continue regular 15-day scheduled inspections."
            ]

        return {
            'feature_effects': feature_effects,
            'positive_contributors': positive_contributors,
            'negative_contributors': negative_contributors,
            'top_feature': positive_contributors[0] if positive_contributors else feature_effects[0],
            'explanation_summary': explanation_summary,
            'recommendations': recommendations,
            'chart_payload_json': json.dumps(chart_payload)
        }

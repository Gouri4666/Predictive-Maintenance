import os
from flask import Flask, render_template, session
from config import Config
from database.db import init_db
from routes.auth import auth_bp
from routes.dashboard import dashboard_bp
from routes.machines import machines_bp
from routes.prediction import prediction_bp
from routes.history import history_bp
from routes.analytics import analytics_bp
from routes.reports import reports_bp
from routes.settings import settings_bp

def create_app():
    """Application Factory Pattern for CNC Predictive Maintenance Web App"""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Auto-initialize database on application startup
    with app.app_context():
        try:
            init_db()
        except Exception as e:
            print(f"[App Warning] Database initialization check: {e}")

    # Register Route Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(machines_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(settings_bp)

    # Template Global Context Processors
    @app.context_processor
    def inject_global_data():
        active_model_info = Config.MODELS_CONFIG.get(Config.DEFAULT_MODEL, {})
        return {
            'app_title': 'CNC Predictive Maintenance System',
            'current_user': session.get('username'),
            'user_role': session.get('role'),
            'user_email': session.get('email'),
            'active_model_name': active_model_info.get('name', 'XGBoost (SMOTE)'),
            'models_config': Config.MODELS_CONFIG
        }

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f"  CNC PREDICTIVE MAINTENANCE SYSTEM — WEB APPLICATION")
    print(f"  Running on: http://127.0.0.1:{port}")
    print(f"  Default Login: admin / admin123")
    print(f"=======================================================\n")
    app.run(host='127.0.0.1', port=port, debug=True)

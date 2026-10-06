from flask import Blueprint, render_template
from routes.auth import login_required
from database.db import query_db
from config import Config

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
@login_required
def index():
    # 1. Summary KPI Metrics
    total_machines = query_db("SELECT COUNT(*) as cnt FROM machines", one=True)['cnt']
    healthy_machines = query_db("SELECT COUNT(*) as cnt FROM machines WHERE status = 'Healthy'", one=True)['cnt']
    at_risk_machines = query_db("SELECT COUNT(*) as cnt FROM machines WHERE status IN ('At Risk', 'Critical')", one=True)['cnt']
    total_predictions = query_db("SELECT COUNT(*) as cnt FROM predictions", one=True)['cnt']
    predicted_failures = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 1", one=True)['cnt']

    # 2. Recent Predictions (last 5)
    recent_predictions = query_db("""
        SELECT p.*, m.machine_name, m.machine_code, m.location
        FROM predictions p
        JOIN machines m ON p.machine_id = m.id
        ORDER BY p.created_at DESC
        LIMIT 6
    """)

    # 3. Chart Data: Outcomes (Normal vs Failure)
    normal_count = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 0", one=True)['cnt']
    failure_count = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 1", one=True)['cnt']

    # 4. Chart Data: Risk Levels
    low_risk = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'Low Risk'", one=True)['cnt']
    med_risk = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'Medium Risk'", one=True)['cnt']
    high_risk = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'High Risk'", one=True)['cnt']

    # 5. Machine Health Snapshot
    machines_summary = query_db("""
        SELECT m.*, 
               (SELECT COUNT(*) FROM predictions p WHERE p.machine_id = m.id) as total_preds,
               (SELECT failure_probability FROM predictions p WHERE p.machine_id = m.id ORDER BY p.created_at DESC LIMIT 1) as last_prob,
               (SELECT risk_level FROM predictions p WHERE p.machine_id = m.id ORDER BY p.created_at DESC LIMIT 1) as last_risk
        FROM machines m
        ORDER BY m.id ASC
    """)

    default_model_info = Config.MODELS_CONFIG.get(Config.DEFAULT_MODEL, {})

    return render_template(
        'dashboard.html',
        total_machines=total_machines,
        healthy_machines=healthy_machines,
        at_risk_machines=at_risk_machines,
        total_predictions=total_predictions,
        predicted_failures=predicted_failures,
        recent_predictions=recent_predictions,
        machines_summary=machines_summary,
        chart_outcomes={'normal': normal_count, 'failure': failure_count},
        chart_risks={'low': low_risk, 'medium': med_risk, 'high': high_risk},
        active_model_name='Predictive Maintenance Model'
    )

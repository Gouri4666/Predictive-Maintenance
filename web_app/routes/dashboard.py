from flask import Blueprint, render_template, session, flash, redirect, url_for, request
from routes.auth import login_required
from database.db import query_db, modify_db
from config import Config
from models.inspection_service import get_machine_inspection_info, update_machine_inspection_schedule

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
@login_required
def index():
    # 1. Base Machine List & Inspection Calculations
    raw_machines = query_db("SELECT * FROM machines ORDER BY id ASC")
    
    inspection_machines = []
    overdue_count = 0
    due_today_count = 0
    due_soon_count = 0
    high_risk_count = 0
    
    for m in raw_machines:
        info = get_machine_inspection_info(m)
        inspection_machines.append(info)
        
        if info['is_overdue']:
            overdue_count += 1
        elif info['is_due_today']:
            due_today_count += 1
        elif info['status_key'] == 'due_soon':
            due_soon_count += 1

        if info['risk_level'] in ('High Risk', 'Critical'):
            high_risk_count += 1

    total_machines = len(inspection_machines)

    # 2. Filter Alerts: Display ONLY machines due on today's date or overdue
    inspection_alerts = [m for m in inspection_machines if m['is_due_today'] or m['is_overdue']]
    inspection_alerts.sort(key=lambda x: (
        0 if x['is_overdue'] else 1,
        0 if x['risk_level'] in ('High Risk', 'Critical') else (1 if x['risk_level'] in ('Medium Risk', 'At Risk') else 2)
    ))

    # 3. Chart Data & Historical Metrics
    total_predictions = query_db("SELECT COUNT(*) as cnt FROM predictions", one=True)['cnt']
    predicted_failures = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 1", one=True)['cnt']
    healthy_machines = sum(1 for m in inspection_machines if m['risk_level'] in ('Healthy', 'Low Risk'))
    at_risk_machines = sum(1 for m in inspection_machines if m['risk_level'] in ('Medium Risk', 'At Risk', 'High Risk', 'Critical'))

    # Calculate fleet failure vs non-failure status for top cards
    latest_preds = query_db("""
        SELECT machine_id, prediction 
        FROM predictions 
        WHERE id IN (
            SELECT MAX(id) FROM predictions GROUP BY machine_id
        )
    """)
    latest_pred_map = {p['machine_id']: p['prediction'] for p in latest_preds}

    failure_machines_count = 0
    non_failure_machines_count = 0
    for m in raw_machines:
        pred = latest_pred_map.get(m['id'])
        if pred is not None:
            if pred == 1 or m.get('status') in ('Critical', 'High Risk'):
                failure_machines_count += 1
            else:
                non_failure_machines_count += 1
        else:
            if m.get('status') in ('Critical', 'High Risk'):
                failure_machines_count += 1
            else:
                non_failure_machines_count += 1

    recent_predictions = query_db("""
        SELECT p.*, m.machine_name, m.machine_code, m.location
        FROM predictions p
        JOIN machines m ON p.machine_id = m.id
        ORDER BY p.created_at DESC
        LIMIT 6
    """)

    normal_count = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 0", one=True)['cnt']
    failure_count = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 1", one=True)['cnt']

    low_risk = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'Low Risk'", one=True)['cnt']
    med_risk = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'Medium Risk'", one=True)['cnt']
    high_risk = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'High Risk'", one=True)['cnt']

    return render_template(
        'dashboard.html',
        total_machines=total_machines,
        failure_machines_count=failure_machines_count,
        non_failure_machines_count=non_failure_machines_count,
        healthy_machines=healthy_machines,
        at_risk_machines=at_risk_machines,
        total_predictions=total_predictions,
        predicted_failures=predicted_failures,
        recent_predictions=recent_predictions,
        inspection_machines=inspection_machines,
        inspection_alerts=inspection_alerts,
        due_today_count=due_today_count,
        due_soon_count=due_soon_count,
        overdue_count=overdue_count,
        high_risk_count=high_risk_count,
        chart_outcomes={'normal': normal_count, 'failure': failure_count},
        chart_risks={'low': low_risk, 'medium': med_risk, 'high': high_risk}
    )

@dashboard_bp.route('/inspections/complete/<int:machine_id>', methods=['POST'])
@login_required
def complete_inspection(machine_id):
    notes = request.form.get('notes', 'Routine inspection completed by operator').strip()
    inspector = session.get('username', 'Operator')
    
    info = update_machine_inspection_schedule(
        machine_id,
        is_manual_completion=True,
        inspector_name=inspector,
        notes=notes
    )
    
    if info:
        flash(f"Inspection completed successfully. Next inspection scheduled for {info['interval_days']} days.", 'success')
    else:
        flash('Machine not found.', 'danger')
        
    return redirect(url_for('dashboard.index'))

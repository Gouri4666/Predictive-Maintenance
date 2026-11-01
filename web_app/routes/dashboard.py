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
    # 1. Base Machine List & Inspection Information
    raw_machines = query_db("SELECT * FROM machines ORDER BY id ASC")
    total_machines = len(raw_machines)

    # 2. Get latest prediction per machine (1 record per unique machine)
    latest_preds = query_db("""
        SELECT p.* 
        FROM predictions p
        INNER JOIN (
            SELECT machine_id, MAX(id) as max_id
            FROM predictions
            GROUP BY machine_id
        ) latest ON p.id = latest.max_id
    """)
    latest_pred_map = {p['machine_id']: p for p in latest_preds}

    # 3. Unique Machine Conditions and Risk Counts (NOT per prediction)
    failure_machines_count = 0
    healthy_machines_count = 0

    unique_machine_risks = {
        'low': 0,
        'medium': 0,
        'high': 0
    }

    inspection_machines = []
    machine_analytics_list = []
    overdue_count = 0
    due_today_count = 0
    due_soon_count = 0

    for m in raw_machines:
        m_id = m['id']
        info = get_machine_inspection_info(m)
        inspection_machines.append(info)

        if info['is_overdue']:
            overdue_count += 1
        elif info['is_due_today']:
            due_today_count += 1
        elif info['status_key'] == 'due_soon':
            due_soon_count += 1

        latest_p = latest_pred_map.get(m_id)

        # Condition for this UNIQUE machine
        is_failure = False
        if latest_p:
            if latest_p['prediction'] == 1 or m.get('status') in ('Critical', 'High Risk'):
                is_failure = True
            m_risk = latest_p.get('risk_level', 'Low Risk')
            failure_prob = float(latest_p.get('failure_probability', 0.0))
        else:
            if m.get('status') in ('Critical', 'High Risk'):
                is_failure = True
                m_risk = 'High Risk'
            elif m.get('status') in ('At Risk', 'Medium Risk'):
                m_risk = 'Medium Risk'
            else:
                m_risk = 'Low Risk'
            failure_prob = 0.0

        if is_failure:
            failure_machines_count += 1
        else:
            healthy_machines_count += 1

        # Categorize unique machine risk level into Low / Moderate / High
        if 'High' in m_risk or 'Critical' in m_risk or m.get('status') in ('Critical', 'High Risk'):
            unique_machine_risks['high'] += 1
            risk_label = 'High Risk'
            risk_badge = 'danger'
        elif 'Med' in m_risk or 'At Risk' in m_risk or 'Moderate' in m_risk or m.get('status') in ('At Risk', 'Medium Risk'):
            unique_machine_risks['medium'] += 1
            risk_label = 'Moderate Risk'
            risk_badge = 'warning'
        else:
            unique_machine_risks['low'] += 1
            risk_label = 'Low Risk'
            risk_badge = 'success'

        # Total historical inferences run for this specific machine
        pred_cnt_row = query_db("SELECT COUNT(*) as cnt FROM predictions WHERE machine_id = %s", (m_id,), one=True)
        total_tests = pred_cnt_row['cnt'] if pred_cnt_row else 0

        machine_analytics_list.append({
            'id': m_id,
            'machine_code': m['machine_code'],
            'machine_name': m.get('machine_name') or m['machine_code'],
            'location': m.get('location') or 'Factory Floor',
            'status': m.get('status', 'Healthy'),
            'is_failure': is_failure,
            'risk_level': risk_label,
            'risk_badge': risk_badge,
            'failure_prob': failure_prob,
            'total_tests': total_tests,
            'next_inspection_due': info['next_inspection_text'],
            'is_overdue': info['is_overdue'],
            'is_due_today': info['is_due_today'],
            'alert_badge': info['alert_badge']
        })

    # Inspection Alerts: ONLY machines due on today's date or overdue
    inspection_alerts = [m for m in inspection_machines if m['is_due_today'] or m['is_overdue']]
    inspection_alerts.sort(key=lambda x: (
        0 if x['is_overdue'] else 1,
        0 if x['risk_level'] in ('High Risk', 'Critical') else (1 if x['risk_level'] in ('Medium Risk', 'At Risk') else 2)
    ))

    # Fleet telemetry sensor averages across all recorded predictions
    fleet_averages = query_db("""
        SELECT 
            ROUND(AVG(vibration_rms), 2) as avg_vibration,
            ROUND(AVG(temperature_motor), 1) as avg_temp,
            ROUND(AVG(current_phase_avg), 1) as avg_current,
            ROUND(AVG(pressure_level), 1) as avg_pressure,
            ROUND(AVG(rpm), 0) as avg_rpm,
            ROUND(AVG(failure_probability), 1) as avg_probability,
            COUNT(id) as total_lifetime_predictions
        FROM predictions
    """, one=True) or {}

    total_predictions = fleet_averages.get('total_lifetime_predictions') or 0

    recent_predictions = query_db("""
        SELECT p.*, m.machine_name, m.machine_code, m.location
        FROM predictions p
        JOIN machines m ON p.machine_id = m.id
        ORDER BY p.created_at DESC
        LIMIT 6
    """)

    # Calculate Time to Inspect / Inspection Schedule distribution across unique machines
    scheduled_count = max(0, total_machines - (overdue_count + due_today_count + due_soon_count))
    inspection_schedule = {
        'overdue': overdue_count,
        'due_today': due_today_count,
        'due_soon': due_soon_count,
        'scheduled': scheduled_count
    }

    return render_template(
        'dashboard.html',
        total_machines=total_machines,
        failure_machines_count=failure_machines_count,
        healthy_machines_count=healthy_machines_count,
        non_failure_machines_count=healthy_machines_count,
        total_predictions=total_predictions,
        recent_predictions=recent_predictions,
        inspection_machines=inspection_machines,
        inspection_alerts=inspection_alerts,
        due_today_count=due_today_count,
        due_soon_count=due_soon_count,
        overdue_count=overdue_count,
        inspection_schedule=inspection_schedule,
        chart_machine_conditions={'healthy': healthy_machines_count, 'failure': failure_machines_count},
        chart_machine_risks=unique_machine_risks,
        machine_analytics_list=machine_analytics_list,
        all_machines=raw_machines
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

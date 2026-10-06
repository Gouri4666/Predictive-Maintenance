import json
from flask import Blueprint, render_template
from routes.auth import login_required
from database.db import query_db

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics')
@login_required
def analytics_dashboard():
    # 1. Total Metrics
    totals = query_db("""
        SELECT 
            COUNT(id) as total_predictions,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as total_failures,
            SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as total_normal,
            SUM(CASE WHEN risk_level = 'High Risk' THEN 1 ELSE 0 END) as total_high_risk,
            AVG(failure_probability) as avg_probability,
            AVG(vibration_rms) as avg_vibration,
            AVG(temperature_motor) as avg_temp,
            AVG(current_phase_avg) as avg_current,
            AVG(pressure_level) as avg_pressure,
            AVG(rpm) as avg_rpm
        FROM predictions
    """, one=True)

    # 2. Risk Distribution
    risk_stats = query_db("""
        SELECT risk_level, COUNT(id) as count
        FROM predictions
        GROUP BY risk_level
    """)
    risk_dict = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}
    for row in risk_stats:
        if row['risk_level'] in risk_dict:
            risk_dict[row['risk_level']] = row['count']

    # 3. Machine-Wise Statistics
    machine_stats = query_db("""
        SELECT 
            m.machine_code,
            m.machine_name,
            COUNT(p.id) as total_tests,
            SUM(CASE WHEN p.prediction = 1 THEN 1 ELSE 0 END) as failures,
            SUM(CASE WHEN p.prediction = 0 THEN 1 ELSE 0 END) as normal,
            ROUND(AVG(p.failure_probability), 1) as avg_prob
        FROM machines m
        LEFT JOIN predictions p ON m.id = p.machine_id
        GROUP BY m.id
        ORDER BY failures DESC, total_tests DESC
    """)

    # 4. Timeline Trend Data (Last 7 days of predictions)
    timeline_stats = query_db("""
        SELECT 
            DATE_FORMAT(created_at, '%b %d') as date_label,
            COUNT(id) as total,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as failures,
            SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as normal
        FROM predictions
        GROUP BY DATE(created_at), DATE_FORMAT(created_at, '%b %d')
        ORDER BY DATE(created_at) ASC
        LIMIT 10
    """)

    # Format JSON strings safely for Chart.js
    timeline_labels = [row['date_label'] for row in timeline_stats]
    timeline_totals = [int(row['total']) for row in timeline_stats]
    timeline_failures = [int(row['failures']) for row in timeline_stats]

    machine_labels = [row['machine_code'] for row in machine_stats]
    machine_fail_counts = [int(row['failures'] or 0) for row in machine_stats]
    machine_norm_counts = [int(row['normal'] or 0) for row in machine_stats]

    all_machines = query_db("SELECT id, machine_code FROM machines ORDER BY machine_code ASC")

    return render_template(
        'analytics.html',
        totals=totals,
        risk_dict=risk_dict,
        machine_stats=machine_stats,
        all_machines=all_machines,
        timeline_labels_json=json.dumps(timeline_labels),
        timeline_totals_json=json.dumps(timeline_totals),
        timeline_failures_json=json.dumps(timeline_failures),
        machine_labels_json=json.dumps(machine_labels),
        machine_fail_counts_json=json.dumps(machine_fail_counts),
        machine_norm_counts_json=json.dumps(machine_norm_counts)
    )

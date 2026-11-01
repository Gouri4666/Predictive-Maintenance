import io
import csv
from datetime import datetime
from flask import Blueprint, render_template, request, Response, make_response, redirect, url_for
from routes.auth import login_required
from database.db import query_db

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/reports')
@login_required
def reports_index():
    return redirect(url_for('dashboard.index') + '#reports')

@reports_bp.route('/reports/export/csv')
@login_required
def export_csv():
    machine_id = request.args.get('machine_id', '').strip()
    risk_level = request.args.get('risk_level', '').strip()

    where_clauses = ["1=1"]
    params = []

    machine_meta = None
    if machine_id and machine_id.isdigit():
        where_clauses.append("p.machine_id = %s")
        params.append(int(machine_id))
        machine_meta = query_db("SELECT machine_code, machine_name FROM machines WHERE id = %s", (int(machine_id),), one=True)

    if risk_level:
        where_clauses.append("p.risk_level = %s")
        params.append(risk_level)

    where_sql = " AND ".join(where_clauses)

    rows = query_db(f"""
        SELECT 
            p.id,
            m.machine_code,
            m.machine_name,
            m.location,
            p.vibration_rms,
            p.temperature_motor,
            p.current_phase_avg,
            p.pressure_level,
            p.rpm,
            CASE WHEN p.prediction = 1 THEN 'FAILURE IMMINENT' ELSE 'NORMAL' END as prediction_label,
            p.failure_probability,
            p.risk_level,
            p.recommendation,
            p.created_at
        FROM predictions p
        JOIN machines m ON p.machine_id = m.id
        WHERE {where_sql}
        ORDER BY p.created_at DESC
    """, tuple(params))

    # Generate CSV stream in-memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Headers
    writer.writerow([
        "Inspection ID", "Machine Code", "Machine Name", "Location",
        "Vibration RMS (mm/s)", "Motor Temp (°C)", "Phase Current (A)", "Pressure (bar)", "RPM",
        "Prediction Status", "Failure Probability (%)", "Risk Level", "Recommendation", "Timestamp"
    ])

    for r in rows:
        writer.writerow([
            r['id'],
            r['machine_code'],
            r['machine_name'],
            r['location'],
            float(r['vibration_rms']),
            float(r['temperature_motor']),
            float(r['current_phase_avg']),
            float(r['pressure_level']),
            float(r['rpm']),
            r['prediction_label'],
            float(r['failure_probability']),
            r['risk_level'],
            r['recommendation'],
            r['created_at'].strftime('%d-%m-%Y %H:%M:%S') if hasattr(r['created_at'], 'strftime') else str(r['created_at'])
        ])

    csv_data = output.getvalue()
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if machine_meta:
        clean_code = str(machine_meta['machine_code']).replace(' ', '_').replace('/', '_')
        filename = f"CNC_Maintenance_Report_{clean_code}_{timestamp_str}.csv"
    else:
        filename = f"CNC_Complete_Fleet_Maintenance_Report_{timestamp_str}.csv"

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-type"] = "text/csv; charset=utf-8"
    return response

@reports_bp.route('/reports/print')
@login_required
def print_report():
    machine_id = request.args.get('machine_id', '').strip()
    risk_level = request.args.get('risk_level', '').strip()

    where_clauses = ["1=1"]
    params = []

    machine_meta = None
    if machine_id and machine_id.isdigit():
        where_clauses.append("p.machine_id = %s")
        params.append(int(machine_id))
        machine_meta = query_db("SELECT * FROM machines WHERE id = %s", (int(machine_id),), one=True)

    if risk_level:
        where_clauses.append("p.risk_level = %s")
        params.append(risk_level)

    where_sql = " AND ".join(where_clauses)

    records = query_db(f"""
        SELECT 
            p.*,
            m.machine_code,
            m.machine_name,
            m.location
        FROM predictions p
        JOIN machines m ON p.machine_id = m.id
        WHERE {where_sql}
        ORDER BY p.created_at DESC
        LIMIT 100
    """, tuple(params))

    return render_template(
        'print_report.html',
        records=records,
        generated_at=datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        machine_meta=machine_meta,
        risk_filter=risk_level
    )

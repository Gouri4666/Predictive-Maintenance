from flask import Blueprint, render_template, request, redirect, url_for, flash
from routes.auth import login_required
from database.db import query_db, modify_db

machines_bp = Blueprint('machines', __name__)

@machines_bp.route('/machines', methods=['GET', 'POST'])
@login_required
def list_machines():
    # Handle Add Machine on the same page
    if request.method == 'POST':
        code = request.form.get('machine_code', '').strip().upper()

        if not code:
            flash('Machine ID is required (e.g. CNC-001).', 'danger')
            return redirect(url_for('machines.list_machines'))

        # Check unique Machine ID
        existing = query_db("SELECT id FROM machines WHERE machine_code = %s", (code,), one=True)
        if existing:
            flash(f"Machine ID '{code}' already exists. Please enter a unique ID.", 'danger')
            return redirect(url_for('machines.list_machines'))

        modify_db(
            "INSERT INTO machines (machine_code, machine_name, status) VALUES (%s, %s, 'Not Evaluated')",
            (code, code)
        )
        flash(f"Machine '{code}' added successfully!", 'success')
        return redirect(url_for('machines.list_machines'))

    # Update any machine with 0 prediction records to 'Not Evaluated' status
    modify_db("""
        UPDATE machines 
        SET status = 'Not Evaluated' 
        WHERE id NOT IN (SELECT DISTINCT machine_id FROM predictions)
    """)

    machines = query_db("""
        SELECT m.id, m.machine_code, m.status, m.created_at,
               COUNT(p.id) as prediction_count,
               MAX(p.created_at) as last_prediction_at,
               (SELECT failure_probability FROM predictions p2 WHERE p2.machine_id = m.id ORDER BY p2.created_at DESC LIMIT 1) as last_probability,
               (SELECT risk_level FROM predictions p2 WHERE p2.machine_id = m.id ORDER BY p2.created_at DESC LIMIT 1) as last_risk,
               (SELECT prediction FROM predictions p2 WHERE p2.machine_id = m.id ORDER BY p2.created_at DESC LIMIT 1) as last_prediction
        FROM machines m
        LEFT JOIN predictions p ON m.id = p.machine_id
        GROUP BY m.id
        ORDER BY m.id ASC
    """)

    all_predictions = query_db("""
        SELECT p.*, m.machine_code
        FROM predictions p
        JOIN machines m ON p.machine_id = m.id
        ORDER BY p.created_at DESC
        LIMIT 300
    """)

    predictions_by_machine = {}
    for p in all_predictions:
        m_id = p['machine_id']
        if m_id not in predictions_by_machine:
            predictions_by_machine[m_id] = []
        predictions_by_machine[m_id].append(p)

    selected_machine_id = request.args.get('machine_id', type=int)

    return render_template(
        'machines.html',
        machines=machines,
        all_predictions=all_predictions,
        predictions_by_machine=predictions_by_machine,
        selected_machine_id=selected_machine_id
    )

@machines_bp.route('/machines/add', methods=['GET', 'POST'])
@login_required
def add_machine():
    # Seamlessly forward to unified machines page
    if request.method == 'POST':
        return list_machines()
    return redirect(url_for('machines.list_machines'))

@machines_bp.route('/machines/<int:machine_id>')
@login_required
def machine_details(machine_id):
    machine = query_db("SELECT * FROM machines WHERE id = %s", (machine_id,), one=True)
    if not machine:
        flash('CNC Machine not found.', 'danger')
        return redirect(url_for('machines.list_machines'))

    predictions = query_db("""
        SELECT * FROM predictions 
        WHERE machine_id = %s 
        ORDER BY created_at DESC
    """, (machine_id,))

    stats = query_db("""
        SELECT 
            AVG(vibration_rms) as avg_vibration,
            AVG(temperature_motor) as avg_temperature,
            AVG(current_phase_avg) as avg_current,
            AVG(pressure_level) as avg_pressure,
            AVG(rpm) as avg_rpm,
            COUNT(id) as total_tests,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as failure_alerts
        FROM predictions
        WHERE machine_id = %s
    """, (machine_id,), one=True)

    return render_template(
        'machine_details.html',
        machine=machine,
        predictions=predictions,
        stats=stats
    )

@machines_bp.route('/machines/delete/<int:machine_id>', methods=['POST'])
@login_required
def delete_machine(machine_id):
    machine = query_db("SELECT machine_code FROM machines WHERE id = %s", (machine_id,), one=True)
    if machine:
        modify_db("DELETE FROM machines WHERE id = %s", (machine_id,))
        flash(f"Machine '{machine['machine_code']}' deleted successfully.", 'info')
    else:
        flash('Machine not found.', 'danger')
    return redirect(url_for('machines.list_machines'))

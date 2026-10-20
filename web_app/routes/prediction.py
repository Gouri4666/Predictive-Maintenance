from decimal import Decimal, InvalidOperation

from flask import Blueprint, render_template, request, flash
from routes.auth import login_required
from database.db import query_db, modify_db
from models.model_loader import ModelLoader
from models.inspection_service import update_machine_inspection_schedule
from models.ale_service import ALEService

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/prediction', methods=['GET', 'POST'])
@login_required
def predict():
    machines = query_db("""
        SELECT m.id, m.machine_code, m.status,
               (SELECT COUNT(*) FROM predictions p WHERE p.machine_id = m.id) as prediction_count
        FROM machines m
        ORDER BY m.id ASC
    """)
    selected_machine_id = request.args.get('machine_id', type=int)
    result = None
    form_data = {}

    if request.method == 'POST':
        machine_id = request.form.get('machine_id', type=int)
        vibration_str = request.form.get('vibration_rms', '').strip()
        temp_str = request.form.get('temperature_motor', '').strip()
        current_str = request.form.get('current_phase_avg', '').strip()
        pressure_str = request.form.get('pressure_level', '').strip()
        rpm_str = request.form.get('rpm', '').strip()

        form_data = {
            'machine_id': machine_id,
            'vibration_rms': vibration_str,
            'temperature_motor': temp_str,
            'current_phase_avg': current_str,
            'pressure_level': pressure_str,
            'rpm': rpm_str
        }

        # 1. Validation: required machine
        if not machine_id:
            flash('Please select a Machine ID.', 'danger')
            return render_template('prediction.html', machines=machines, selected_machine_id=selected_machine_id, form_data=form_data, result=result)

        # 2. Validation: required fields
        if not all([vibration_str, temp_str, current_str, pressure_str, rpm_str]):
            flash('All 5 sensor parameters are required.', 'danger')
            return render_template('prediction.html', machines=machines, selected_machine_id=machine_id, form_data=form_data, result=result)

        # 3. Validation: numeric values with at most two decimal places
        try:
            numeric_values = [
                Decimal(vibration_str),
                Decimal(temp_str),
                Decimal(current_str),
                Decimal(pressure_str),
                Decimal(rpm_str),
            ]
            if any(not value.is_finite() for value in numeric_values):
                raise InvalidOperation
            if any(value != value.quantize(Decimal('0.01')) for value in numeric_values):
                raise ValueError('More than two decimal places')

            vibration, temperature, current, pressure, rpm = map(float, numeric_values)
        except (InvalidOperation, ValueError):
            flash('Sensor parameters must be valid numbers with at most 2 decimal places.', 'danger')
            return render_template('prediction.html', machines=machines, selected_machine_id=machine_id, form_data=form_data, result=result)

        # 4. Run ML Prediction using XGBoost
        try:
            pred_output = ModelLoader.predict(
                vibration_rms=vibration,
                temperature_motor=temperature,
                current_phase_avg=current,
                pressure_level=pressure,
                rpm=rpm
            )
        except Exception as e:
            flash(f"Prediction failed: {str(e)}", 'danger')
            return render_template('prediction.html', machines=machines, selected_machine_id=machine_id, form_data=form_data, result=result)

        # 6. Save to Database
        pred_id = modify_db("""
            INSERT INTO predictions (
                machine_id, vibration_rms, temperature_motor,
                current_phase_avg, pressure_level, rpm, prediction,
                failure_probability, risk_level, recommendation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            machine_id,
            vibration,
            temperature,
            current,
            pressure,
            rpm,
            pred_output['prediction'],
            pred_output['failure_probability'],
            pred_output['risk_level'],
            pred_output['recommendation']
        ))

        # 7. Update Machine Status
        if pred_output['risk_level'] == 'High Risk':
            new_status = 'Critical'
        elif pred_output['risk_level'] == 'Medium Risk':
            new_status = 'At Risk'
        else:
            new_status = 'Healthy'

        modify_db("UPDATE machines SET status = %s WHERE id = %s", (new_status, machine_id))
        update_machine_inspection_schedule(machine_id, risk_level=new_status)

        # Lookup machine code for display
        machine_meta = query_db("SELECT machine_code FROM machines WHERE id = %s", (machine_id,), one=True)
        machine_code = machine_meta['machine_code'] if machine_meta else f"ID {machine_id}"

        # Compute Accumulated Local Effects (ALE) Explanation
        ale_explanation = ALEService.explain_prediction(
            inputs=pred_output['inputs'],
            failure_probability=pred_output['failure_probability'],
            risk_level=pred_output['risk_level']
        )

        # Attach output result for right-side display
        result = {
            'pred_id': pred_id,
            'machine_id': machine_id,
            'machine_code': machine_code,
            'prediction': pred_output['prediction'],
            'prediction_label': 'LIKELY TO FAIL' if pred_output['prediction'] == 1 else 'UNLIKELY TO FAIL',
            'failure_probability': pred_output['failure_probability'],
            'risk_level': pred_output['risk_level'],
            'inputs': pred_output['inputs'],
            'ale': ale_explanation
        }

    return render_template(
        'prediction.html',
        machines=machines,
        selected_machine_id=selected_machine_id or (form_data.get('machine_id') if form_data else None),
        form_data=form_data,
        result=result
    )

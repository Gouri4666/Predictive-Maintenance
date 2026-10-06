from flask import Blueprint, render_template, request, flash
from routes.auth import login_required
from database.db import query_db, modify_db
from models.model_loader import ModelLoader

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/prediction', methods=['GET', 'POST'])
@login_required
def predict():
    machines = query_db("SELECT id, machine_code, status FROM machines ORDER BY id ASC")
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

        # 3. Validation: numeric values
        try:
            vibration = float(vibration_str)
            temperature = float(temp_str)
            current = float(current_str)
            pressure = float(pressure_str)
            rpm = float(rpm_str)
        except ValueError:
            flash('All sensor parameters must be valid numbers.', 'danger')
            return render_template('prediction.html', machines=machines, selected_machine_id=machine_id, form_data=form_data, result=result)

        # 4. Physical range bounds
        errors = []
        if not (0.01 <= vibration <= 30.0): errors.append("Vibration RMS should be between 0.01 and 30 mm/s.")
        if not (5.0 <= temperature <= 160.0): errors.append("Motor Temperature should be between 5 and 160 °C.")
        if not (0.5 <= current <= 120.0): errors.append("Phase Current should be between 0.5 and 120 A.")
        if not (1.0 <= pressure <= 150.0): errors.append("Pressure Level should be between 1 and 150 bar.")
        if not (50.0 <= rpm <= 7000.0): errors.append("Spindle RPM should be between 50 and 7000 RPM.")

        if errors:
            for err in errors:
                flash(err, 'warning')
            return render_template('prediction.html', machines=machines, selected_machine_id=machine_id, form_data=form_data, result=result)

        # 5. Run ML Prediction using strictly the champion SMOTE-XGBoost model
        try:
            pred_output = ModelLoader.predict(
                vibration_rms=vibration,
                temperature_motor=temperature,
                current_phase_avg=current,
                pressure_level=pressure,
                rpm=rpm,
                model_key='xgboost'  # Fixed to champion SMOTE-XGBoost
            )
        except Exception as e:
            flash(f"Prediction failed: {str(e)}", 'danger')
            return render_template('prediction.html', machines=machines, selected_machine_id=machine_id, form_data=form_data, result=result)

        # 6. Save to Database
        pred_id = modify_db("""
            INSERT INTO predictions (
                machine_id, model_used, vibration_rms, temperature_motor,
                current_phase_avg, pressure_level, rpm, prediction,
                failure_probability, risk_level, recommendation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            machine_id,
            'Predictive Model',
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
        if pred_output['prediction'] == 1 or pred_output['risk_level'] == 'High Risk':
            new_status = 'Critical'
        elif pred_output['risk_level'] == 'Medium Risk':
            new_status = 'At Risk'
        else:
            new_status = 'Healthy'

        modify_db("UPDATE machines SET status = %s WHERE id = %s", (new_status, machine_id))

        # Lookup machine code for display
        machine_meta = query_db("SELECT machine_code FROM machines WHERE id = %s", (machine_id,), one=True)
        machine_code = machine_meta['machine_code'] if machine_meta else f"ID {machine_id}"

        # Attach output result for right-side display
        result = {
            'pred_id': pred_id,
            'machine_id': machine_id,
            'machine_code': machine_code,
            'prediction': pred_output['prediction'],
            'prediction_label': 'LIKELY TO FAIL' if pred_output['prediction'] == 1 else 'UNLIKELY TO FAIL',
            'failure_probability': pred_output['failure_probability'],
            'risk_level': pred_output['risk_level'],
            'inputs': pred_output['inputs']
        }
        flash('Prediction generated and saved successfully!', 'success')

    return render_template(
        'prediction.html',
        machines=machines,
        selected_machine_id=selected_machine_id or (form_data.get('machine_id') if form_data else None),
        form_data=form_data,
        result=result
    )

@prediction_bp.route('/prediction/result/<int:pred_id>')
@login_required
def prediction_result(pred_id):
    prediction = query_db("""
        SELECT p.*, m.machine_code
        FROM predictions p
        JOIN machines m ON p.machine_id = m.id
        WHERE p.id = %s
    """, (pred_id,), one=True)

    if not prediction:
        flash('Prediction record not found.', 'danger')
        return render_template('prediction.html')

    return render_template(
        'prediction_result.html',
        prediction=prediction
    )

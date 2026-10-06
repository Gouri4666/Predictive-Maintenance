from flask import Blueprint, redirect, url_for, request
from routes.auth import login_required

history_bp = Blueprint('history', __name__)

@history_bp.route('/history')
@login_required
def history_list():
    """
    Prediction history is now integrated into the unified CNC Machines page.
    Redirect smoothly to preserve backwards compatibility.
    """
    machine_id = request.args.get('machine_id')
    if machine_id:
        return redirect(url_for('machines.list_machines', machine_id=machine_id))
    return redirect(url_for('machines.list_machines'))

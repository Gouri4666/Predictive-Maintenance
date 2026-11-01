import os
from datetime import datetime, timedelta
from database.db import query_db, modify_db
from config import Config

def get_inspection_interval_days(status_or_risk):
    """Return the configured inspection interval in days based on machine risk level."""
    return Config.INSPECTION_INTERVALS.get(status_or_risk, 15)

def update_machine_inspection_schedule(machine_id, risk_level=None, is_manual_completion=False, inspector_name="Operator", notes=""):
    """
    Recalculates last_inspection_at and next_inspection_due for a machine.
    Sets last_inspection_at = now and next_inspection_due = now + interval_days.
    """
    machine = query_db("SELECT * FROM machines WHERE id = %s", (machine_id,), one=True)
    if not machine:
        return None

    current_risk = risk_level or machine.get('status') or 'Not Evaluated'
    interval_days = get_inspection_interval_days(current_risk)
    now = datetime.now()

    last_insp = now
    next_due = now + timedelta(days=interval_days)

    if is_manual_completion:
        prev_risk = machine.get('status') or 'Not Evaluated'
        modify_db("""
            INSERT INTO inspections (machine_id, inspector_name, previous_risk_level, current_risk_level, notes, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (machine_id, inspector_name, prev_risk, current_risk, notes, now))

    modify_db("""
        UPDATE machines 
        SET last_inspection_at = %s, next_inspection_due = %s 
        WHERE id = %s
    """, (last_insp, next_due, machine_id))

    return {
        'last_inspection_at': last_insp,
        'next_inspection_due': next_due,
        'interval_days': interval_days,
        'current_risk': current_risk
    }

def get_machine_inspection_info(machine):
    """
    Calculates inspection status and alert info for a machine.
    """
    now = datetime.now()
    risk = machine.get('status') or 'Not Evaluated'
    interval_days = get_inspection_interval_days(risk)

    last_insp = machine.get('last_inspection_at')
    next_due = machine.get('next_inspection_due')

    if not next_due:
        if not last_insp:
            # Not yet inspected -> Due today!
            next_due = now
        else:
            next_due = last_insp + timedelta(days=interval_days)

    is_overdue = next_due.date() < now.date()
    is_due_today = (next_due.date() == now.date()) or (not last_insp)
    is_due_soon = (now.date() < next_due.date() <= (now + timedelta(days=7)).date())

    if is_overdue:
        next_inspection_text = "Overdue"
        status_key = 'overdue'
        status_label = 'Overdue'
        alert_badge = 'danger'
    elif is_due_today:
        next_inspection_text = "Today"
        status_key = 'due_today'
        status_label = 'Due Today'
        alert_badge = 'danger' if risk in ('High Risk', 'Critical') else 'warning'
    elif is_due_soon:
        next_inspection_text = next_due.strftime('%d-%m-%Y')
        status_key = 'due_soon'
        status_label = 'Due Soon'
        alert_badge = 'warning'
    else:
        next_inspection_text = next_due.strftime('%d-%m-%Y')
        status_key = 'scheduled'
        status_label = 'Healthy'
        alert_badge = 'success'

    return {
        'id': machine['id'],
        'machine_code': machine['machine_code'],
        'machine_name': machine.get('machine_name') or machine['machine_code'],
        'location': machine.get('location') or '',
        'risk_level': risk,
        'interval_days': interval_days,
        'last_inspection_at': last_insp,
        'next_inspection_due': next_due,
        'is_overdue': is_overdue,
        'is_due_today': is_due_today,
        'status_key': status_key,
        'status_label': status_label,
        'alert_badge': alert_badge,
        'next_inspection_text': next_inspection_text
    }

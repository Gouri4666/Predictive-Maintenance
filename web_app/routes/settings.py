from flask import Blueprint, render_template, request, session, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from routes.auth import login_required
from database.db import query_db, modify_db

settings_bp = Blueprint('settings', __name__)

@settings_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings_index():
    user = query_db("SELECT id, username, email, role, created_at FROM users WHERE id = %s", (session.get('user_id'),), one=True)
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'change_password':
            current_pw = request.form.get('current_password', '').strip()
            new_pw = request.form.get('new_password', '').strip()
            confirm_pw = request.form.get('confirm_password', '').strip()

            user_auth = query_db("SELECT password_hash FROM users WHERE id = %s", (session.get('user_id'),), one=True)
            
            if not check_password_hash(user_auth['password_hash'], current_pw):
                flash('Current password is incorrect.', 'danger')
            elif len(new_pw) < 6:
                flash('New password must be at least 6 characters long.', 'danger')
            elif new_pw != confirm_pw:
                flash('New passwords do not match.', 'danger')
            else:
                new_hash = generate_password_hash(new_pw)
                modify_db("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, session.get('user_id')))
                flash('Password updated successfully.', 'success')

        return redirect(url_for('settings.settings_index'))

    return render_template('settings.html', user=user)

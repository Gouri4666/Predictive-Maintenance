/**
 * CNC Predictive Maintenance System — Interactive Scripts
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Auto-dismiss flash alerts after 6 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function (alert) {
        setTimeout(function () {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 6000);
    });


    // 3. Confirmation dialog for deletion actions
    const deleteForms = document.querySelectorAll('.confirm-delete');
    deleteForms.forEach(form => {
        form.addEventListener('submit', function (e) {
            const confirmed = confirm("Are you sure you want to delete this CNC record? This will remove its historical prediction logs permanently.");
            if (!confirmed) {
                e.preventDefault();
            }
        });
    });

    // 4. Mobile / Tablet Sidebar Toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', function () {
            sidebar.classList.toggle('show-mobile');
        });

        // Close sidebar on click outside on mobile
        document.addEventListener('click', function(e) {
            if (window.innerWidth < 992) {
                if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
                    sidebar.classList.remove('show-mobile');
                }
            }
        });
    }
});

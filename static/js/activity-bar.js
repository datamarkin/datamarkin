/**
 * Activity bar — polls /api/tasks and shows background task status.
 * Loaded from base.html, runs on every page.
 */
(function () {
    var bar = document.getElementById('activity-bar');
    var toggle = document.getElementById('activity-toggle');
    var panel = document.getElementById('activity-panel');
    var list = document.getElementById('activity-list');
    var label = document.getElementById('activity-label');
    var dot = bar ? bar.querySelector('.activity-dot') : null;

    if (!bar) return;

    var panelOpen = false;
    var prevTaskIds = null;  // null = first poll (skip toasts)

    toggle.addEventListener('click', function (e) {
        e.stopPropagation();
        panelOpen = !panelOpen;
        panel.style.display = panelOpen ? '' : 'none';
    });

    document.addEventListener('click', function (e) {
        if (panelOpen && !panel.contains(e.target) && e.target !== toggle) {
            panelOpen = false;
            panel.style.display = 'none';
        }
    });

    // Event delegation for cancel buttons (avoids re-attaching listeners every poll)
    list.addEventListener('click', function (e) {
        var btn = e.target.closest('.activity-task-cancel');
        if (!btn) return;
        e.stopPropagation();
        var taskId = btn.getAttribute('data-task-id');
        fetch('/api/tasks/' + taskId + '/cancel', { method: 'POST' });
    });

    function statusDotClass(status) {
        if (status === 'running') return 'is-running';
        if (status === 'queued') return 'is-queued';
        if (status === 'failed') return 'is-failed';
        if (status === 'cancelled') return 'is-cancelled';
        return 'is-done';
    }

    function esc(str) {
        var d = document.createElement('div');
        d.textContent = str || '';
        return d.innerHTML;
    }

    function renderTask(task) {
        var pct = Math.round((task.progress || 0) * 100);
        var canCancel = task.status === 'running' || task.status === 'queued';

        var html = '<div class="activity-task" data-task-id="' + task.id + '">';
        html += '<div class="activity-task-header">';
        html += '<span class="activity-task-dot ' + statusDotClass(task.status) + '"></span>';
        html += '<span class="activity-task-label">' + esc(task.label) + '</span>';
        if (canCancel) {
            html += '<button class="activity-task-cancel" data-task-id="' + task.id + '" title="Cancel">&times;</button>';
        }
        html += '</div>';

        if (task.status === 'running' || task.status === 'queued') {
            html += '<div class="activity-task-progress">';
            html += '<div class="activity-task-bar" style="width:' + pct + '%"></div>';
            html += '</div>';
            if (task.detail) {
                html += '<div class="activity-task-detail">' + esc(task.detail) + '</div>';
            }
        } else if (task.status === 'failed' && task.error) {
            html += '<div class="activity-task-detail activity-task-error">' + esc(task.error) + '</div>';
        }

        html += '</div>';
        return html;
    }

    function poll() {
        fetch('/api/tasks/')
            .then(function (r) { return r.json(); })
            .then(function (tasks) {
                var active = tasks.filter(function (t) {
                    return t.status === 'running' || t.status === 'queued';
                });

                // Show/hide bar
                if (tasks.length > 0) {
                    bar.style.display = '';
                    if (active.length > 0) {
                        dot.classList.add('is-active');
                        var n = active.length;
                        label.textContent = n + (n === 1 ? ' task' : ' tasks');
                    } else {
                        dot.classList.remove('is-active');
                        label.textContent = '';
                    }
                } else {
                    bar.style.display = 'none';
                    if (panelOpen) {
                        panelOpen = false;
                        panel.style.display = 'none';
                    }
                }

                // Toast on newly completed tasks (skip first poll)
                var newPrev = {};
                tasks.forEach(function (t) {
                    if (t.status === 'done' || t.status === 'failed') {
                        newPrev[t.id] = true;
                        if (prevTaskIds && !prevTaskIds[t.id] && typeof window.showToast === 'function') {
                            if (t.status === 'done') {
                                window.showToast(t.label + ' completed', 'success');
                            } else {
                                window.showToast(t.label + ' failed: ' + (t.error || 'Unknown error'), 'error');
                            }
                        }
                    }
                });
                prevTaskIds = newPrev;

                // Render panel
                if (panelOpen) {
                    var html = '';
                    tasks.forEach(function (t) { html += renderTask(t); });
                    if (!html) html = '<div class="activity-task-empty">No tasks</div>';
                    list.innerHTML = html;
                }
            })
            .catch(function () { /* ignore poll errors */ });
    }

    var POLL_ACTIVE = 3000;
    var POLL_IDLE = 15000;

    function schedulePoll() {
        var hasActive = dot && dot.classList.contains('is-active');
        setTimeout(function () { poll(); schedulePoll(); }, hasActive ? POLL_ACTIVE : POLL_IDLE);
    }

    poll();
    schedulePoll();

    // Warn on browser tab close (dev mode)
    window.addEventListener('beforeunload', function (e) {
        if (dot && dot.classList.contains('is-active')) {
            e.preventDefault();
            e.returnValue = 'Tasks are still running.';
        }
    });
})();

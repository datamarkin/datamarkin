/**
 * Auto-annotate batch handler with progress polling.
 * Used by project.html.
 *
 * Expects PAGE_CONFIG: { projectId }
 */
(function () {
    var projectId = PAGE_CONFIG.projectId;

    document.getElementById('auto-annotate-btn').addEventListener('click', function () {
        var btn = this;
        btn.classList.add('is-loading');
        btn.disabled = true;

        fetch('/api/falcon/auto_annotate_batch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({project_id: projectId, target: 'all'})
        })
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
            if (data.error) {
                alert(data.error);
                btn.classList.remove('is-loading');
                btn.disabled = false;
                return;
            }
            if (data.status === 'done' && data.total === 0) {
                btn.classList.remove('is-loading');
                btn.disabled = false;
                return;
            }
            document.getElementById('auto-annotate-banner').style.display = '';
            var pollErrors = 0;
            var poll = setInterval(function () {
                fetch('/api/falcon/auto_annotate_batch_status')
                    .then(function (r) { return r.json(); })
                    .then(function (s) {
                        pollErrors = 0;
                        document.getElementById('auto-annotate-progress').textContent = s.current + ' / ' + s.total;
                        var pct = s.total ? (s.current / s.total * 100) : 0;
                        document.getElementById('auto-annotate-bar').value = pct;
                        if (s.status === 'done' || s.status === 'error') {
                            clearInterval(poll);
                            if (s.status === 'error') alert('Auto-annotate error: ' + (s.error || 'Unknown'));
                            location.reload();
                        }
                    })
                    .catch(function (e) {
                        console.error('Poll error:', e);
                        if (++pollErrors >= 5) {
                            clearInterval(poll);
                            alert('Lost connection to server');
                        }
                    });
            }, 2000);
        })
        .catch(function (e) {
            console.error('Auto annotate batch failed:', e);
            btn.classList.remove('is-loading');
            btn.disabled = false;
        });
    });
})();

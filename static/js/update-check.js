/**
 * App update checker.
 * Polls GitHub releases once per session and shows a toast when an update is available.
 * Depends on: toast.js (window.showToast)
 */
(function() {
    if (sessionStorage.getItem('update-check-done')) return;
    sessionStorage.setItem('update-check-done', '1');

    var container = document.getElementById('download-toasts');

    fetch('/api/update-check')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (!data.available) return;
            if (sessionStorage.getItem('dismiss-update-' + data.version)) return;

            var el = document.createElement('div');
            el.style.cssText = 'pointer-events:auto;background:#f7ecd9;border:1px solid #e8d5b5;border-radius:8px;padding:10px 14px;min-width:280px;max-width:360px;font-size:13px;font-family:inherit;box-shadow:0 2px 8px rgba(0,0,0,0.1);';
            el.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:start;gap:8px;margin-bottom:6px;">'
                + '<strong>Datamarkin ' + data.version + ' available</strong>'
                + '<span class="update-close" style="cursor:pointer;opacity:0.5;font-size:16px;line-height:1;">&times;</span></div>'
                + '<div class="update-actions" style="display:flex;gap:8px;align-items:center;">'
                + (data.download_url
                    ? '<button class="update-dl-btn" style="background:#a08050;color:#fff;border:none;border-radius:4px;padding:4px 12px;font-size:12px;cursor:pointer;">Download</button>'
                    : '')
                + '<a href="' + data.url + '" target="_blank" style="color:#a08050;font-size:12px;">View release</a></div>';

            el.querySelector('.update-close').onclick = function() {
                el.style.display = 'none';
                sessionStorage.setItem('dismiss-update-' + data.version, '1');
            };

            var dlBtn = el.querySelector('.update-dl-btn');
            if (dlBtn) {
                dlBtn.onclick = function() {
                    dlBtn.disabled = true;
                    dlBtn.textContent = 'Downloading\u2026';
                    fetch('/api/update-download', { method: 'POST' })
                        .then(function(r) { return r.json(); })
                        .then(function(res) {
                            if (res.ok) {
                                dlBtn.textContent = 'Downloaded';
                                dlBtn.style.background = '#6a9a5a';
                                showToast('Update downloaded. Check Finder to install.', 'success', 5000);
                            } else {
                                dlBtn.textContent = 'Download';
                                dlBtn.disabled = false;
                                showToast('Download failed: ' + (res.error || 'Unknown'), 'error', 5000);
                            }
                        })
                        .catch(function() {
                            dlBtn.textContent = 'Download';
                            dlBtn.disabled = false;
                            showToast('Download failed. Check your connection.', 'error', 5000);
                        });
                };
            }

            container.appendChild(el);
        })
        .catch(function() { /* offline — do nothing */ });
})();

/**
 * Toast notifications + SSE download progress.
 * Loaded on every page via base.html.
 *
 * Exposes:
 *   window.showToast(message, type, duration)
 *   window.startDownloadSSE()
 */
(function() {
    var container = document.getElementById('download-toasts');

    window.showToast = function(message, type, duration) {
        type = type || 'info';
        duration = duration !== undefined ? duration : 5000;
        var colors = {
            error:   { bg: '#fdf0f0', border: '#e8c5c5', bar: '#c44', icon: '' },
            info:    { bg: '#f7ecd9', border: '#e8d5b5', bar: '#a08050', icon: '' },
            success: { bg: '#f0f8f0', border: '#c5e0c5', bar: '#6a9a5a', icon: '' },
        };
        var c = colors[type] || colors.info;
        var el = document.createElement('div');
        el.style.cssText = 'pointer-events:auto;border-radius:8px;padding:10px 14px;min-width:280px;max-width:400px;font-size:13px;font-family:inherit;box-shadow:0 2px 8px rgba(0,0,0,0.1);transition:opacity 0.3s;background:' + c.bg + ';border:1px solid ' + c.border + ';';
        el.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:start;gap:8px;"><span>' + escapeHtml(message) + '</span><span style="cursor:pointer;opacity:0.5;font-size:16px;line-height:1;" class="toast-close">&times;</span></div>';
        el.querySelector('.toast-close').onclick = function() { dismiss(el); };
        container.appendChild(el);
        if (duration > 0) {
            setTimeout(function() { dismiss(el); }, duration);
        }
    };

    function dismiss(el) {
        el.style.opacity = '0';
        setTimeout(function() { el.remove(); }, 300);
    }

    function escapeHtml(s) {
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    // ── SSE download progress (lazy — only connects when downloads are active) ─
    var toasts = {};
    var evtSource = null;

    function connectSSE() {
        if (evtSource) return;
        evtSource = new EventSource('/api/downloads/stream');
        evtSource.onmessage = function(e) {
            var downloads = JSON.parse(e.data);
            var activeNames = new Set();
            downloads.forEach(function(dl) {
                activeNames.add(dl.name);
                if (!toasts[dl.name]) {
                    toasts[dl.name] = createDlToast(dl.name);
                    container.appendChild(toasts[dl.name]);
                }
                updateDlToast(toasts[dl.name], dl);
            });
            Object.keys(toasts).forEach(function(name) {
                if (!activeNames.has(name) && toasts[name]) {
                    toasts[name].remove();
                    delete toasts[name];
                }
            });
        };
        evtSource.onerror = function() {
            evtSource.close();
            evtSource = null;
        };
    }

    function createDlToast(name) {
        var el = document.createElement('div');
        el.style.cssText = 'pointer-events:auto;background:#f7ecd9;border:1px solid #e8d5b5;border-radius:8px;padding:10px 14px;min-width:280px;max-width:360px;font-size:13px;font-family:inherit;box-shadow:0 2px 8px rgba(0,0,0,0.1);transition:opacity 0.3s;';
        el.innerHTML = '<div style="display:flex;justify-content:space-between;margin-bottom:6px;"><strong class="dl-name"></strong><span class="dl-pct" style="color:#8a7a5a;"></span></div><div style="background:#e8d5b5;border-radius:4px;height:6px;overflow:hidden;"><div class="dl-bar" style="background:#a08050;height:100%;width:0%;transition:width 0.3s;border-radius:4px;"></div></div><div class="dl-status" style="margin-top:4px;font-size:11px;color:#8a7a5a;"></div>';
        return el;
    }

    function updateDlToast(el, dl) {
        el.querySelector('.dl-name').textContent = dl.name;
        el.querySelector('.dl-pct').textContent = dl.pct + '%';
        el.querySelector('.dl-bar').style.width = dl.pct + '%';
        if (dl.status === 'downloading') {
            el.querySelector('.dl-status').textContent = 'Downloading...';
        } else if (dl.status === 'ready') {
            el.querySelector('.dl-status').textContent = 'Complete';
            el.querySelector('.dl-bar').style.width = '100%';
            el.querySelector('.dl-bar').style.background = '#6a9a5a';
            setTimeout(function() { dismiss(el); delete toasts[dl.name]; }, 3000);
        } else if (dl.status === 'error') {
            el.querySelector('.dl-status').textContent = 'Error: ' + (dl.error || 'Unknown');
            el.querySelector('.dl-bar').style.background = '#c44';
        }
    }

    // Only connect if downloads are already active
    fetch('/api/downloads/status')
        .then(function(r) { return r.json(); })
        .then(function(downloads) {
            if (downloads.some(function(d) { return d.status !== 'idle'; })) connectSSE();
        })
        .catch(function() {});

    window.startDownloadSSE = connectSSE;
})();

/**
 * Training modal — run list, detail view, polling, metrics charts.
 * Used by project.html.
 *
 * Expects PAGE_CONFIG: { projectId, activeTrainingId }
 */
(function () {
    var projectId = PAGE_CONFIG.projectId;
    var modal = document.getElementById('training-modal');
    var listView = document.getElementById('training-list-view');
    var detailView = document.getElementById('training-detail-view');
    var backBtn = document.getElementById('training-back-btn');
    var titleEl = document.getElementById('training-modal-title');
    var startBtn = document.getElementById('training-start-btn');
    var stopBtn = document.getElementById('training-stop-btn');
    var currentId = null;
    var epochTimer = null;
    var liveTimer = null;
    var lossHistory = [];

    // Open modal
    document.getElementById('train-btn').addEventListener('click', function () {
        modal.classList.add('is-active');
        if (activeTrainingId) startPolling(activeTrainingId);
    });

    // Close
    document.getElementById('training-modal-close').addEventListener('click', closeModal);
    modal.querySelector('.modal-background').addEventListener('click', closeModal);

    function closeModal() {
        modal.classList.remove('is-active');
        stopPolling();
    }

    // Back to list
    backBtn.addEventListener('click', showList);

    function showList() {
        listView.style.display = '';
        detailView.style.display = 'none';
        backBtn.style.display = 'none';
        stopBtn.style.display = 'none';
        titleEl.textContent = 'Training runs';
        currentId = null;
        lossHistory = [];
        stopPolling();
        if (activeTrainingId) startPolling(activeTrainingId);
    }

    function showDetail(trainingId) {
        listView.style.display = 'none';
        detailView.style.display = 'block';
        backBtn.style.display = '';
        currentId = trainingId;
        lossHistory = [];
        fetchAndRender(trainingId);
    }

    // Row click → detail
    document.querySelectorAll('.training-row').forEach(function (row) {
        row.addEventListener('click', function () {
            stopPolling();
            showDetail(row.dataset.trainingId);
        });
    });

    // ── Polling ──────────────────────────────────────────────────────
    var activeTrainingId = PAGE_CONFIG.activeTrainingId || '';

    function startPolling(id) {
        stopPolling();
        epochTimer = setInterval(function () {
            fetch('/api/training/' + id)
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    updateListRow(data);
                    if (currentId === id) renderDetail(data);
                    if (data.status !== 'running' && data.status !== 'pending') {
                        stopPolling();
                        activeTrainingId = '';
                        startBtn.disabled = false;
                        if (currentId === id) stopBtn.style.display = 'none';
                    }
                });
        }, 5000);
        liveTimer = setInterval(function () {
            if (currentId !== id) return;
            fetch('/api/training/' + id + '/live')
                .then(function (r) { return r.json(); })
                .then(function (live) {
                    if (live.batch_loss === undefined) return;
                    lossHistory.push(live.batch_loss);
                    var el = document.getElementById('tm-chart-loss');
                    if (el) drawLineChart(el, lossHistory, '#4a90d9');
                    var lbl = document.getElementById('tm-live-loss');
                    if (lbl) lbl.textContent = live.batch_loss.toFixed(4);
                });
        }, 2000);
    }

    function stopPolling() {
        if (epochTimer) { clearInterval(epochTimer); epochTimer = null; }
        if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    }

    function fetchAndRender(id) {
        fetch('/api/training/' + id)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                renderDetail(data);
                if (data.status === 'running' || data.status === 'pending') {
                    stopBtn.style.display = '';
                    startPolling(id);
                } else {
                    stopBtn.style.display = 'none';
                }
            });
    }

    // ── Detail renderer ───────────────────────────────────────────────
    function renderDetail(t) {
        var cfg = parse(t.config);
        var metrics = parse(t.metrics);
        var progress = parse(t.progress);
        var history = metrics.history || [];
        var created = (t.created_at || '').slice(0, 10);
        var isLive = t.status === 'running' || t.status === 'pending';

        var statusColor = {
            running: '#48c774', pending: '#ffdd57', done: '#48c774',
            failed: '#f14668', stopped: '#aaa'
        };
        var dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + (statusColor[t.status] || '#aaa') + ';margin-right:6px' + (isLive ? ';animation:pulse 1.5s infinite' : '') + '"></span>';
        var statusLine = '<div style="margin-bottom:12px">' + dot + '<strong>' + esc(t.status) + '</strong><span class="is-size-7 has-text-grey ml-2">' + created + '</span></div>';

        // Epoch progress bar
        var progressHTML = '';
        if (isLive) {
            var epoch = progress.epoch || 0;
            var total = cfg.epochs || 0;
            var pct = total > 0 ? Math.round(epoch / total * 100) : 0;
            progressHTML =
                '<div class="mb-4">' +
                '<div style="display:flex;justify-content:space-between;margin-bottom:4px">' +
                '<span class="is-size-7">Epoch ' + epoch + ' / ' + total + '</span>' +
                '<span class="is-size-7">' + pct + '%</span></div>' +
                '<progress class="progress is-small is-link" value="' + pct + '" max="100"></progress></div>';
        }

        // Accuracy panel
        var mapVal = progress.map !== undefined ? progress.map : (history.length > 0 ? (history[history.length - 1].map || 0) : null);
        var prevMapVal = history.length >= 2 ? (history[history.length - 2].map || 0) : null;
        var accuracyHTML = '';
        if (mapVal !== null || history.length > 0 || t.status === 'done') {
            var pct100 = mapVal !== null ? (mapVal * 100) : 0;
            var barColor = pct100 < 15 ? '#e76f51' : pct100 < 40 ? '#f4a261' : '#2a9d8f';
            var trendHTML = '';
            if (prevMapVal !== null && mapVal !== null) {
                var delta = (mapVal - prevMapVal) * 100;
                var arrow = delta >= 0 ? '↑' : '↓';
                var dColor = delta >= 0 ? '#2a9d8f' : '#e76f51';
                trendHTML = ' <span style="color:' + dColor + ';font-size:0.8rem">' + arrow + ' ' + Math.abs(delta).toFixed(1) + '% this epoch</span>';
            }
            var bestMapVal = metrics.best_mAP !== undefined ? metrics.best_mAP : mapVal;
            accuracyHTML =
                '<div class="mb-4" style="background:#faf4eb;border-radius:8px;padding:14px 16px">' +
                '<p class="is-size-7 has-text-grey mb-2" style="text-transform:uppercase;letter-spacing:.05em">Model Accuracy</p>' +
                '<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px">' +
                '<span style="font-size:2rem;font-weight:700;color:' + barColor + '">' + pct100.toFixed(1) + '%</span>' +
                trendHTML +
                (t.status === 'done' && bestMapVal ? '<span class="is-size-7 has-text-grey" style="margin-left:auto">Best: ' + (bestMapVal * 100).toFixed(1) + '%</span>' : '') +
                '</div>' +
                '<div style="background:#e8ddd0;border-radius:4px;height:6px;overflow:hidden">' +
                '<div style="width:' + Math.min(100, pct100) + '%;height:100%;background:' + barColor + ';border-radius:4px;transition:width .6s ease"></div></div>' +
                '<p class="is-size-7 has-text-grey mt-2">How accurately the model detects objects in images it hasn\'t seen during training (mAP@50).</p></div>';
        }

        // Training curves
        var hasEpochLoss = history.length > 0 && history[0].loss !== undefined;
        var hasEpochMap = history.length > 0 && history[0].map !== undefined;
        var hasLiveLoss = lossHistory.length > 0;
        var chartsHTML = '';
        if (hasLiveLoss || hasEpochLoss || hasEpochMap) {
            var lossSection = '';
            if (hasLiveLoss || hasEpochLoss) {
                lossSection =
                    '<div style="margin-bottom:12px">' +
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">' +
                    '<span class="is-size-7 has-text-grey">Loss <span style="font-size:0.7em;color:#aaa">(lower is better)</span></span>' +
                    (isLive ? '<span class="is-size-7" id="tm-live-loss" style="color:#4a90d9;font-weight:600"></span>' : '') +
                    '</div>' +
                    '<svg id="tm-chart-loss" width="100%" height="72" style="display:block;overflow:visible"></svg></div>';
            }
            var mapSection = '';
            if (hasEpochMap) {
                mapSection =
                    '<div>' +
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">' +
                    '<span class="is-size-7 has-text-grey">Detection accuracy per epoch <span style="font-size:0.7em;color:#aaa">(higher is better)</span></span>' +
                    '</div>' +
                    '<svg id="tm-chart-map" width="100%" height="72" style="display:block;overflow:visible"></svg></div>';
            }
            chartsHTML =
                '<div class="mb-4" style="background:#faf4eb;border-radius:8px;padding:14px 16px">' +
                '<p class="is-size-7 has-text-grey mb-3" style="text-transform:uppercase;letter-spacing:.05em">Training Curves</p>' +
                lossSection + mapSection + '</div>';
        }

        // Technical details
        var techHTML = '';
        if (history.length > 0) {
            var last = history[history.length - 1];
            var techParts = [];
            if (last.loss !== undefined) techParts.push('Loss: <strong>' + last.loss.toFixed(4) + '</strong>');
            if (last.map_50_95 !== undefined) techParts.push('mAP@50:95: <strong>' + (last.map_50_95 * 100).toFixed(1) + '%</strong>');
            if (last.map_75 !== undefined) techParts.push('mAP@75: <strong>' + (last.map_75 * 100).toFixed(1) + '%</strong>');
            if (last.recall !== undefined) techParts.push('Recall: <strong>' + (last.recall * 100).toFixed(1) + '%</strong>');
            if (last.f1 !== undefined) techParts.push('F1: <strong>' + (last.f1 * 100).toFixed(1) + '%</strong>');
            if (last.precision !== undefined) techParts.push('Precision: <strong>' + (last.precision * 100).toFixed(1) + '%</strong>');
            if (last.ema_map_50 !== undefined) techParts.push('EMA mAP@50: <strong>' + (last.ema_map_50 * 100).toFixed(1) + '%</strong>');
            var keys = Object.keys(last);
            var tableRows = history.map(function (e, i) {
                return '<tr><td>' + (i + 1) + '</td>' +
                    keys.map(function (k) {
                        var v = e[k];
                        return '<td>' + (typeof v === 'number' ? v.toFixed(4) : esc(String(v === null || v === undefined ? '—' : v))) + '</td>';
                    }).join('') + '</tr>';
            }).join('');
            techHTML =
                '<details class="mb-3"><summary class="is-size-7" style="cursor:pointer;color:#888">Technical details (' + history.length + ' epoch' + (history.length !== 1 ? 's' : '') + ')</summary>' +
                '<div class="mt-2 mb-3 is-size-7" style="line-height:2">' + techParts.join(' &nbsp;·&nbsp; ') + '</div>' +
                '<div style="overflow:auto;max-height:160px">' +
                '<table class="table is-narrow is-fullwidth is-size-7" style="white-space:nowrap">' +
                '<thead><tr><th>#</th>' + keys.map(function (k) { return '<th>' + esc(k) + '</th>'; }).join('') + '</tr></thead>' +
                '<tbody>' + tableRows + '</tbody></table></div></details>';
        }

        // Error
        var errorHTML = '';
        if (t.status === 'failed' && t.error) {
            errorHTML =
                '<div class="mb-3">' +
                '<p class="is-size-7 has-text-danger mb-1">Error</p>' +
                '<pre class="is-size-7" style="max-height:140px;overflow:auto;background:#fff5f5;padding:8px;border-radius:4px;white-space:pre-wrap">' +
                esc(t.error) + '</pre></div>';
        }

        // Model path
        var modelHTML = '';
        if (t.status === 'done' && t.model_path) {
            modelHTML =
                '<div class="mb-3">' +
                '<p class="is-size-7 has-text-grey mb-1">Saved model</p>' +
                '<p class="is-size-7" style="word-break:break-all;color:#555">' + esc(t.model_path) + '</p></div>';
        }

        // Config summary
        var cfgHTML =
            '<details class="mb-3"><summary class="is-size-7" style="cursor:pointer;color:#888">Configuration</summary>' +
            '<p class="is-size-7 mt-1">' +
            'Model: <strong>' + (cfg.model_size || '—') + '</strong>' +
            ' · Resolution: <strong>' + (cfg.resolution || '—') + 'px</strong>' +
            ' · Epochs: <strong>' + (cfg.epochs || '—') + '</strong>' +
            ' · Batch: <strong>' + (cfg.batch_size || '—') + '</strong>' +
            ' · LR: <strong>' + (cfg.lr || '—') + '</strong>' +
            '</p></details>';

        detailView.innerHTML = statusLine + progressHTML + accuracyHTML + chartsHTML + errorHTML + modelHTML + techHTML + cfgHTML;

        // Draw charts after DOM update
        var lossEl = document.getElementById('tm-chart-loss');
        var mapEl = document.getElementById('tm-chart-map');
        if (lossEl) {
            var lossVals = lossHistory.length > 0
                ? lossHistory
                : history.map(function (e) { return e.loss !== undefined ? e.loss : null; });
            drawLineChart(lossEl, lossVals, '#4a90d9');
        }
        if (mapEl) {
            var mapVals = history.map(function (e) { return e.map !== undefined ? e.map : null; });
            drawLineChart(mapEl, mapVals, '#2a9d8f');
        }
    }

    function drawLineChart(el, values, color) {
        var nums = values.map(function (v) { return v === null || v === undefined ? NaN : +v; });
        var valid = nums.filter(function (v) { return !isNaN(v); });
        if (valid.length < 1) {
            el.innerHTML = '<text x="50%" y="50%" text-anchor="middle" font-size="10" fill="#aaa">waiting for data…</text>';
            return;
        }
        var W = 300, H = 60;
        var mn = Math.min.apply(null, valid), mx = Math.max.apply(null, valid);
        var pad = (mx - mn) * 0.15 || 0.01;
        var pts = [];
        var n = nums.length;
        nums.forEach(function (v, i) {
            if (isNaN(v)) return;
            pts.push((n > 1 ? i / (n - 1) * W : W / 2).toFixed(1) + ',' + (H - (v - mn + pad) / (mx - mn + 2 * pad) * H).toFixed(1));
        });
        el.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        el.setAttribute('preserveAspectRatio', 'none');
        var fillPts = pts[0].split(',')[0] + ',' + H + ' ' + pts.join(' ') + ' ' + pts[pts.length - 1].split(',')[0] + ',' + H;
        el.innerHTML =
            '<polygon points="' + fillPts + '" fill="' + color + '" fill-opacity="0.08"/>' +
            '<polyline points="' + pts.join(' ') + '" fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>' +
            '<circle cx="' + pts[pts.length - 1].split(',')[0] + '" cy="' + pts[pts.length - 1].split(',')[1] + '" r="2.5" fill="' + color + '"/>';
    }

    function updateListRow(t) {
        var row = document.querySelector('.training-row[data-training-id="' + t.id + '"]');
        if (!row) return;
        var dot = row.querySelector('.training-status-dot');
        var colorMap = {
            running: '#48c774', pending: '#ffdd57', done: '#48c774',
            failed: '#f14668', stopped: '#aaa'
        };
        if (dot) {
            dot.style.background = colorMap[t.status] || '#aaa';
            dot.style.animation = (t.status === 'running' || t.status === 'pending') ? 'pulse 1.5s infinite' : '';
        }
        var epochEl = row.querySelector('.training-epoch');
        if (epochEl) {
            var progress = parse(t.progress);
            var cfg = parse(t.config);
            if ((t.status === 'running' || t.status === 'pending') && progress.epoch) {
                epochEl.textContent = ' (epoch ' + progress.epoch + '/' + (cfg.epochs || '?') + ')';
            }
        }
    }

    function parse(v) {
        if (!v) return {};
        if (typeof v === 'object') return v;
        try { return JSON.parse(v); } catch (e) { return {}; }
    }

    function esc(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // ── Start / Stop ────────────────────────────────────────────────
    startBtn.addEventListener('click', function () {
        if (startBtn.disabled) return;
        startBtn.disabled = true;
        startBtn.textContent = 'Starting…';
        fetch('/api/training/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({project_id: projectId}),
        }).then(function (r) { return r.json(); })
        .then(function (d) {
            if (d.training_id) {
                activeTrainingId = d.training_id;
                showDetail(d.training_id);
            } else {
                alert(d.error || 'Failed to start training');
                startBtn.disabled = false;
                startBtn.textContent = '+ Start Training';
            }
        }).catch(function () {
            startBtn.disabled = false;
            startBtn.textContent = '+ Start Training';
        });
    });

    stopBtn.addEventListener('click', function () {
        if (!currentId) return;
        fetch('/api/training/' + currentId + '/stop', {method: 'POST'})
            .then(function (r) { return r.json(); })
            .then(function () { fetchAndRender(currentId); });
    });
})();

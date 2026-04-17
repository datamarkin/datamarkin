/**
 * Project settings modal — labels, pipeline, config, split, file upload.
 * Used by project.html.
 *
 * Expects PAGE_CONFIG: { projectId, preprocessing, augmentation, configuration }
 * Depends on: pipeline-builder.js, file-upload.js, Sortable.js
 */

// ─── Global label helpers (called from onclick attributes) ────────────
function addLabelRow() {
    var list = document.getElementById('labels-list');
    var div = document.createElement('div');
    div.className = 'label-row-wrapper mb-2';
    div.innerHTML =
        '<div class="field has-addons mt-0 mb-0">' +
        '<div class="control is-expanded"><input class="input label-name" type="text" placeholder="Label name" value=""></div>' +
        '<div class="control"><input class="label-color" type="color" value="#3498db" style="height:40px;width:50px;border:1px solid #dbdbdb;border-radius:4px;cursor:pointer;"></div>' +
        '<div class="control"><button type="button" class="button is-light" onclick="this.closest(\'.label-row-wrapper\').remove()">×</button></div>' +
        '</div>' +
        '<div class="keypoint-editor pl-4 pt-1" style="display:none">' +
        '<div class="keypoints-list"></div>' +
        '<button type="button" class="button is-small is-light mt-1" onclick="addKeypointToLabel(this)">+ Add keypoint</button>' +
        '<div class="skeleton-editor mt-2" style="display:none">' +
        '<p class="is-size-7 has-text-grey mb-1">Skeleton connections</p>' +
        '<div class="skeleton-connections-list"></div>' +
        '<button type="button" class="button is-small is-light mt-1" onclick="addSkeletonConnection(this)">+ Add connection</button>' +
        '</div>' +
        '</div>';
    list.appendChild(div);
}

function addKeypointToLabel(btn) {
    var kpList = btn.closest('.keypoint-editor').querySelector('.keypoints-list');
    var kpRow = document.createElement('div');
    kpRow.className = 'field has-addons mt-1 mb-0 keypoint-row';
    kpRow.innerHTML =
        '<div class="control is-expanded"><input class="input is-small" type="text" placeholder="Keypoint name"></div>' +
        '<div class="control"><input type="color" value="#e74c3c" style="height:32px;width:40px;border:1px solid #dbdbdb;border-radius:4px;cursor:pointer;"></div>' +
        '<div class="control"><button type="button" class="button is-small is-light" onclick="this.closest(\'.keypoint-row\').remove()">×</button></div>';
    kpList.appendChild(kpRow);
}

function addSkeletonConnection(btn) {
    var wrapper = btn.closest('.label-row-wrapper');
    var kpNames = [];
    wrapper.querySelectorAll('.keypoints-list .keypoint-row').forEach(function (kpRow, i) {
        var name = kpRow.querySelector('input[type="text"]').value.trim() || ('Point ' + i);
        kpNames.push({id: i, name: name});
    });
    if (kpNames.length < 2) return;

    var connList = btn.closest('.skeleton-editor').querySelector('.skeleton-connections-list');
    var row = document.createElement('div');
    row.className = 'field has-addons mt-1 mb-0 skeleton-row';

    var fromSelect = document.createElement('select');
    fromSelect.className = 'skeleton-from';
    var toSelect = document.createElement('select');
    toSelect.className = 'skeleton-to';
    kpNames.forEach(function (kp) {
        var o1 = document.createElement('option');
        o1.value = kp.id; o1.textContent = kp.name;
        fromSelect.appendChild(o1);
        var o2 = document.createElement('option');
        o2.value = kp.id; o2.textContent = kp.name;
        toSelect.appendChild(o2);
    });

    row.innerHTML =
        '<div class="control"><div class="select is-small"></div></div>' +
        '<div class="control"><span class="button is-small is-static">&rarr;</span></div>' +
        '<div class="control"><div class="select is-small"></div></div>' +
        '<div class="control"><button type="button" class="button is-small is-light" onclick="this.closest(\'.skeleton-row\').remove()">&times;</button></div>';
    row.querySelectorAll('.select')[0].appendChild(fromSelect);
    row.querySelectorAll('.select')[1].appendChild(toSelect);
    connList.appendChild(row);
}

(function () {
    var projectId = PAGE_CONFIG.projectId;

    // --- Settings modal ---
    var modal = document.getElementById('project-settings-modal');

    document.getElementById('settings-btn').addEventListener('click', function () {
        modal.classList.add('is-active');
    });
    modal.querySelector('.modal-background').addEventListener('click', function () {
        modal.classList.remove('is-active');
    });
    document.getElementById('settings-modal-close').addEventListener('click', function () {
        modal.classList.remove('is-active');
    });

    // --- Tab switching ---
    modal.querySelectorAll('.settings-nav-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            modal.querySelectorAll('.settings-nav-btn').forEach(function (b) {
                b.classList.remove('is-active');
            });
            btn.classList.add('is-active');
            modal.querySelectorAll('.settings-panel').forEach(function (p) {
                p.classList.remove('is-active');
            });
            document.getElementById('panel-' + btn.dataset.panel).classList.add('is-active');
        });
    });

    // --- Pipeline builder init ---
    buildCatalog(document.getElementById('pp-catalog'), PP_CATALOG);
    buildCatalog(document.getElementById('aug-catalog'), AUG_CATALOG);
    initCatalogClicks(document.getElementById('pp-catalog'), document.getElementById('pp-pipeline'), PP_CATALOG, false);
    initCatalogClicks(document.getElementById('aug-catalog'), document.getElementById('aug-pipeline'), AUG_CATALOG, true);
    initSearch(document.getElementById('pp-search'), document.getElementById('pp-catalog'));
    initSearch(document.getElementById('aug-search'), document.getElementById('aug-catalog'));
    initPipelineSortable(document.getElementById('pp-pipeline'), PP_CATALOG, false);
    initPipelineSortable(document.getElementById('aug-pipeline'), AUG_CATALOG, true);
    initCatalogSortable(document.getElementById('pp-catalog'), false);
    initCatalogSortable(document.getElementById('aug-catalog'), true);
    loadPipeline(document.getElementById('pp-pipeline'), PAGE_CONFIG.preprocessing, PP_CATALOG, false);
    loadPipeline(document.getElementById('aug-pipeline'), PAGE_CONFIG.augmentation, AUG_CATALOG, true);

    // --- Save project settings ---
    function collectAndSave(feedbackEl) {
        var name = document.querySelector('#panel-settings input[type="text"]').value.trim();
        var desc = document.querySelector('#panel-settings textarea').value;
        if (!name) {
            feedbackEl.textContent = 'Name is required';
            return;
        }
        var labels = [];
        document.querySelectorAll('#labels-list .label-row-wrapper').forEach(function (row, i) {
            var labelName = row.querySelector('.label-name').value.trim();
            var color = row.querySelector('.label-color').value.replace('#', '');
            if (!labelName) return;

            var labelObj = {id: i, name: labelName, color: color};

            var keypoints = [];
            row.querySelectorAll('.keypoints-list .keypoint-row').forEach(function (kpRow, kpIndex) {
                var kpName = kpRow.querySelector('input[type="text"]').value.trim();
                var kpColor = kpRow.querySelector('input[type="color"]').value.replace('#', '');
                if (kpName) keypoints.push({id: kpIndex, name: kpName, color: kpColor});
            });

            if (keypoints.length > 0) {
                labelObj.keypoints = keypoints;
                var skeleton = [];
                row.querySelectorAll('.skeleton-connections-list .skeleton-row').forEach(function (connRow) {
                    var from = parseInt(connRow.querySelector('.skeleton-from').value, 10);
                    var to = parseInt(connRow.querySelector('.skeleton-to').value, 10);
                    if (!isNaN(from) && !isNaN(to) && from !== to) skeleton.push([from, to]);
                });
                labelObj.skeleton = skeleton;
            }

            labels.push(labelObj);
        });
        fetch('/project/' + projectId + '/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name, description: desc, labels: labels}),
        }).then(function (r) {
            return r.json();
        }).then(function (data) {
            feedbackEl.textContent = data.ok ? 'Saved' : (data.error || 'Error');
            setTimeout(function () {
                feedbackEl.textContent = '';
            }, 2500);
        });
    }

    document.getElementById('save-settings-btn').addEventListener('click', function () {
        collectAndSave(document.getElementById('save-settings-feedback'));
    });
    document.getElementById('save-labels-btn').addEventListener('click', function () {
        collectAndSave(document.getElementById('save-labels-feedback'));
    });

    // --- File upload ---
    initFileUpload(projectId);

    // --- Training configuration ---
    (function () {
        var cfg = PAGE_CONFIG.configuration;
        if (cfg.model_size) document.getElementById('cfg-model-size').value = cfg.model_size;
        if (cfg.resolution) document.getElementById('cfg-resolution').value = String(cfg.resolution);
        if (cfg.epochs) document.getElementById('cfg-epochs').value = cfg.epochs;
        if (cfg.batch_size) document.getElementById('cfg-batch-size').value = cfg.batch_size;
        if (cfg.lr) document.getElementById('cfg-lr').value = cfg.lr;
        if (cfg.early_stopping === false) document.getElementById('cfg-early-stopping').checked = false;
        if (cfg.early_stopping_patience) document.getElementById('cfg-patience').value = cfg.early_stopping_patience;

        function togglePatience() {
            document.getElementById('cfg-patience-field').style.display =
                document.getElementById('cfg-early-stopping').checked ? '' : 'none';
        }

        document.getElementById('cfg-early-stopping').addEventListener('change', togglePatience);
        togglePatience();

        // --- Split sliders ---
        var sliderTrain = document.getElementById('slider-train');
        var sliderVal = document.getElementById('slider-val');

        function updateSplitUI() {
            var train = parseInt(sliderTrain.value);
            var maxVal = Math.max(5, 95 - train);
            sliderVal.max = maxVal;
            if (parseInt(sliderVal.value) > maxVal) sliderVal.value = maxVal;
            var val = parseInt(sliderVal.value);
            var test = 100 - train - val;
            if (test < 0) {
                sliderVal.value = 100 - train - 5;
                val = parseInt(sliderVal.value);
                test = 100 - train - val;
            }
            document.getElementById('lbl-train').textContent = train + '%';
            document.getElementById('lbl-val').textContent = val + '%';
            document.getElementById('lbl-test').textContent = test + '%';
            document.getElementById('split-bar-train').style.width = train + '%';
            document.getElementById('split-bar-val').style.width = val + '%';
            document.getElementById('split-bar-test').style.width = test + '%';
        }

        sliderTrain.addEventListener('input', updateSplitUI);
        sliderVal.addEventListener('input', updateSplitUI);

        // Restore saved split ratios
        if (cfg.split_train) sliderTrain.value = Math.round(cfg.split_train * 100);
        if (cfg.split_val) sliderVal.value = Math.round(cfg.split_val * 100);
        updateSplitUI();

        document.getElementById('apply-split-btn').addEventListener('click', function () {
            var fb = document.getElementById('apply-split-feedback');
            var train = parseInt(sliderTrain.value) / 100;
            var val = parseInt(sliderVal.value) / 100;
            var test = Math.round((1 - train - val) * 100) / 100;
            fetch('/project/' + projectId + '/apply-split', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({train: train, val: val, test: test}),
            }).then(function (r) {
                return r.json();
            }).then(function (d) {
                if (d.ok) {
                    fb.textContent = 'Applied — train:' + d.counts.train + ' val:' + d.counts.valid + ' test:' + d.counts.test;
                } else {
                    fb.textContent = d.error || 'Error';
                }
                setTimeout(function () {
                    fb.textContent = '';
                }, 3500);
            });
        });

        document.getElementById('save-config-btn').addEventListener('click', function () {
            var fb = document.getElementById('save-config-feedback');
            var payload = {
                model_size: document.getElementById('cfg-model-size').value,
                resolution: parseInt(document.getElementById('cfg-resolution').value),
                epochs: parseInt(document.getElementById('cfg-epochs').value),
                batch_size: parseInt(document.getElementById('cfg-batch-size').value),
                lr: parseFloat(document.getElementById('cfg-lr').value),
                early_stopping: document.getElementById('cfg-early-stopping').checked,
                early_stopping_patience: parseInt(document.getElementById('cfg-patience').value),
                split_train: parseInt(sliderTrain.value) / 100,
                split_val: parseInt(sliderVal.value) / 100,
            };
            fetch('/project/' + projectId + '/configuration', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            }).then(function (r) {
                return r.json();
            }).then(function (d) {
                fb.textContent = d.ok ? 'Saved' : (d.error || 'Error');
                setTimeout(function () {
                    fb.textContent = '';
                }, 2500);
            });
        });
    })();
})();

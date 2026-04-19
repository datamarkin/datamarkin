/**
 * Studio playground — image upload, model/workflow inference, results display.
 * Expects PAGE_CONFIG: { mode, trainingId, workflowJson }
 */
(function () {
    var MODE = PAGE_CONFIG.mode;
    var TRAINING_ID = PAGE_CONFIG.trainingId;
    var WORKFLOW_JSON = PAGE_CONFIG.workflowJson;

    var dropZone = document.getElementById('drop-zone');
    var fileInput = document.getElementById('file-input');
    var runBtn = document.getElementById('run-btn');
    var runBtnText = document.getElementById('run-btn-text');
    var threshRange = document.getElementById('threshold-range');
    var threshDisp = document.getElementById('threshold-display');
    var errorMsg = document.getElementById('error-msg');
    var emptyState = document.getElementById('empty-state');
    var dropZoneText = document.getElementById('drop-zone-text');

    if (threshRange) {
        threshRange.addEventListener('input', function () {
            threshDisp.textContent = parseFloat(threshRange.value).toFixed(2);
        });
    }

    dropZone.addEventListener('click', function () { fileInput.click(); });
    dropZone.addEventListener('dragover', function (e) { e.preventDefault(); dropZone.classList.add('is-dragover'); });
    dropZone.addEventListener('dragleave', function () { dropZone.classList.remove('is-dragover'); });

    function showError(msg) { errorMsg.textContent = msg; errorMsg.classList.remove('is-hidden'); }
    function hideError() { errorMsg.textContent = ''; errorMsg.classList.add('is-hidden'); }
    function escapeHtml(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
    function buildKeypointMap(det) { var m = {}; if (det.keypoints) det.keypoints.forEach(function (kp) { m[kp.name] = kp; }); return m; }

    // ========================================================================
    // MODEL MODE — batch
    // ========================================================================
    if (MODE === 'model') {
        var imagesCol = document.getElementById('batch-images');
        var batchResults = document.getElementById('batch-results');
        var resultsTbody = document.getElementById('results-tbody');
        var resultsSummary = document.getElementById('results-summary');
        var csvWrap = document.getElementById('csv-download-wrap');

        var batchItems = []; // [{file, fileName, dataURI, annotatedImage, detections, cardEl}]
        var isRunning = false;
        var activeIndex = -1;
        var keypointNames = null; // discovered from first detection with keypoints
        var isVideoMode = false;
        var videoFile = null;

        dropZone.addEventListener('drop', function (e) {
            e.preventDefault();
            dropZone.classList.remove('is-dragover');
            if (e.dataTransfer.files.length > 0) loadFiles(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', function () {
            if (fileInput.files.length > 0) loadFiles(fileInput.files);
            fileInput.value = '';
        });

        function loadFiles(fileList) {
            if (isRunning) return;
            var allFiles = Array.from(fileList);
            var videos = allFiles.filter(function (f) { return f.type.startsWith('video/'); });
            var images = allFiles.filter(function (f) { return f.type.startsWith('image/'); });

            // Reset common state
            batchItems = [];
            activeIndex = -1;
            imagesCol.innerHTML = '';
            resultsTbody.innerHTML = '';
            batchResults.style.display = 'none';
            emptyState.style.display = '';
            csvWrap.style.display = 'none';
            resultsSummary.textContent = '';

            if (videos.length > 0) {
                isVideoMode = true;
                videoFile = videos[0];
                document.getElementById('video-controls').style.display = '';
                dropZoneText.textContent = videoFile.name;
                return;
            }

            isVideoMode = false;
            videoFile = null;
            document.getElementById('video-controls').style.display = 'none';

            if (!images.length) return;

            images.forEach(function (file) {
                var item = { file: file, fileName: file.name, dataURI: null, annotatedImage: null, detections: null, cardEl: null };
                batchItems.push(item);
                var reader = new FileReader();
                reader.onload = function (e) { item.dataURI = e.target.result; };
                reader.readAsDataURL(file);
            });

            dropZoneText.textContent = images.length === 1 ? images[0].name : images.length + ' images selected';
        }

        function highlightImage(idx) {
            activeIndex = idx;
            var cards = imagesCol.querySelectorAll('.batch-image-card');
            for (var i = 0; i < cards.length; i++) cards[i].classList.toggle('is-active', i === idx);
            var rows = resultsTbody.querySelectorAll('tr');
            var firstMatch = null;
            for (var j = 0; j < rows.length; j++) {
                var match = parseInt(rows[j].dataset.imageIndex) === idx;
                rows[j].classList.toggle('is-image-active', match);
                if (match && !firstMatch) firstMatch = rows[j];
            }
            if (firstMatch) firstMatch.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        // ---- Video inference ----
        function runVideo() {
            if (!videoFile) { showError('Please drop a video first.'); return; }
            if (isRunning) return;
            hideError();
            isRunning = true;

            var frameSkip = parseInt(document.getElementById('frame-skip').value) || 5;
            var video = document.createElement('video');
            video.muted = true;
            video.preload = 'auto';
            var objectUrl = URL.createObjectURL(videoFile);
            video.src = objectUrl;

            video.addEventListener('error', function () {
                showError('Could not load video file.');
                isRunning = false;
                runBtn.disabled = false;
                runBtnText.textContent = 'Run';
                URL.revokeObjectURL(objectUrl);
            });

            video.addEventListener('loadedmetadata', function () {
                var duration = video.duration;
                var fps = 30;
                var timeStep = frameSkip / fps;
                var totalSteps = Math.ceil(duration / timeStep);

                emptyState.style.display = 'none';
                batchResults.style.display = '';
                imagesCol.innerHTML = '';
                resultsTbody.innerHTML = '';
                keypointNames = null;
                var thead = resultsTbody.closest('table').querySelector('thead tr');
                var BASE_COL = 5;
                while (thead.children.length > BASE_COL) thead.removeChild(thead.lastChild);
                csvWrap.style.display = 'none';
                resultsSummary.textContent = '';
                runBtn.disabled = true;
                batchItems = [];

                // Single frame display
                var frameImg = document.createElement('img');
                frameImg.style.cssText = 'display:block;width:100%;border-radius:6px;';
                imagesCol.appendChild(frameImg);
                var progressEl = document.createElement('p');
                progressEl.className = 'is-size-7 has-text-grey mt-2 has-text-centered';
                imagesCol.appendChild(progressEl);

                var canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                var ctx2d = canvas.getContext('2d');

                var currentTime = 0;
                var processed = 0;
                var globalRow = 0;

                function formatTs(sec) {
                    var m = Math.floor(sec / 60);
                    var s = sec % 60;
                    return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1);
                }

                function seekAndExtract(t, cb) {
                    function handler() {
                        video.removeEventListener('seeked', handler);
                        ctx2d.drawImage(video, 0, 0);
                        canvas.toBlob(function (blob) { cb(blob); }, 'image/jpeg', 0.85);
                    }
                    video.addEventListener('seeked', handler);
                    video.currentTime = Math.min(t, duration - 0.001);
                }

                function processNext() {
                    if (currentTime >= duration || !isRunning) { finish(); return; }

                    seekAndExtract(currentTime, function (blob) {
                        if (!blob) { currentTime += timeStep; processNext(); return; }

                        processed++;
                        var ts = formatTs(currentTime);
                        runBtnText.textContent = 'Frame ' + processed + '/' + totalSteps + '\u2026';
                        progressEl.textContent = processed + ' / ~' + totalSteps + ' frames';

                        var frameItem = { file: null, fileName: ts, dataURI: null, annotatedImage: null, detections: null, cardEl: null };
                        batchItems.push(frameItem);
                        var idx = batchItems.length - 1;

                        var formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');
                        formData.append('training_id', TRAINING_ID);
                        formData.append('threshold', threshRange.value);

                        fetch('/api/predict/run', { method: 'POST', body: formData })
                            .then(function (resp) {
                                return resp.json().then(function (data) {
                                    if (!resp.ok) throw new Error(data.error || 'Inference failed');
                                    return data;
                                });
                            })
                            .then(function (data) {
                                frameItem.detections = data.detections;
                                frameImg.src = data.annotated_image;

                                // Dynamic keypoint columns
                                if (!keypointNames && data.detections) {
                                    var kpDet = data.detections.find(function (d) { return d.keypoints && d.keypoints.length; });
                                    if (kpDet) {
                                        keypointNames = kpDet.keypoints.map(function (kp) { return kp.name; });
                                        var frag = document.createDocumentFragment();
                                        keypointNames.forEach(function (name) {
                                            var thX = document.createElement('th'); thX.textContent = name + '-x'; frag.appendChild(thX);
                                            var thY = document.createElement('th'); thY.textContent = name + '-y'; frag.appendChild(thY);
                                        });
                                        thead.appendChild(frag);
                                    }
                                }

                                // Table rows
                                var rowFrag = document.createDocumentFragment();
                                (data.detections || []).forEach(function (det) {
                                    globalRow++;
                                    var row = document.createElement('tr');
                                    row.dataset.imageIndex = idx;
                                    var coords = det.bbox.map(function (v) { return Math.round(v); }).join(', ');
                                    var html =
                                        '<td>' + globalRow + '</td>' +
                                        '<td>' + escapeHtml(ts) + '</td>' +
                                        '<td>' + escapeHtml(det.class_name || 'unknown') + '</td>' +
                                        '<td>' + (det.confidence ? Math.round(det.confidence * 100) + '%' : '\u2014') + '</td>' +
                                        '<td style="font-family:monospace">' + coords + '</td>';
                                    if (keypointNames) {
                                        var kpMap = buildKeypointMap(det);
                                        keypointNames.forEach(function (name) {
                                            var kp = kpMap[name];
                                            html += '<td style="font-family:monospace">' + (kp ? Math.round(kp.x) : '\u2014') + '</td>';
                                            html += '<td style="font-family:monospace">' + (kp ? Math.round(kp.y) : '\u2014') + '</td>';
                                        });
                                    }
                                    row.innerHTML = html;
                                    rowFrag.appendChild(row);
                                });
                                resultsTbody.appendChild(rowFrag);

                                currentTime += timeStep;
                                processNext();
                            })
                            .catch(function () {
                                currentTime += timeStep;
                                processNext();
                            });
                    });
                }

                function finish() {
                    URL.revokeObjectURL(objectUrl);
                    isRunning = false;
                    runBtnText.textContent = 'Run';
                    runBtn.disabled = false;
                    resultsSummary.textContent = globalRow + ' detection' + (globalRow !== 1 ? 's' : '') + ' across ' + processed + ' frame' + (processed !== 1 ? 's' : '');
                    if (globalRow > 0) csvWrap.style.display = '';
                }

                processNext();
            });
        }

        // ---- Run ----
        runBtn.addEventListener('click', function () {
            if (isVideoMode) { runVideo(); return; }
            if (!batchItems.length) { showError('Please drop images first.'); return; }
            if (isRunning) return;
            hideError();
            isRunning = true;
            activeIndex = -1;

            var total = batchItems.length;
            var completed = 0;
            var globalRow = 0;

            batchItems.forEach(function (item) { item.annotatedImage = null; item.detections = null; item.cardEl = null; });
            imagesCol.innerHTML = '';
            resultsTbody.innerHTML = '';
            keypointNames = null;
            var thead = resultsTbody.closest('table').querySelector('thead tr');
            var BASE_COLUMN_COUNT = 5; // #, File, Class, Confidence, BBox
            while (thead.children.length > BASE_COLUMN_COUNT) thead.removeChild(thead.lastChild);
            csvWrap.style.display = 'none';
            resultsSummary.textContent = '';
            emptyState.style.display = 'none';
            batchResults.style.display = '';
            runBtn.disabled = true;

            function processNext() {
                if (completed >= total) {
                    isRunning = false;
                    runBtnText.textContent = 'Run';
                    runBtn.disabled = false;
                    var n = 0;
                    batchItems.forEach(function (it) { if (it.detections) n += it.detections.length; });
                    resultsSummary.textContent = n + ' detection' + (n !== 1 ? 's' : '') + ' across ' + total + ' image' + (total !== 1 ? 's' : '');
                    if (n > 0) csvWrap.style.display = '';
                    return;
                }

                var idx = completed;
                var item = batchItems[idx];
                runBtnText.textContent = 'Running ' + (idx + 1) + '/' + total + '\u2026';

                var card = document.createElement('div');
                card.className = 'batch-image-card is-processing';
                var img = document.createElement('img');
                img.src = item.dataURI;
                img.alt = item.fileName;
                card.appendChild(img);
                var nameEl = document.createElement('div');
                nameEl.className = 'batch-image-name';
                nameEl.textContent = item.fileName;
                card.appendChild(nameEl);
                imagesCol.appendChild(card);
                item.cardEl = card;

                (function (capturedIdx) {
                    card.addEventListener('click', function () { highlightImage(capturedIdx); });
                })(idx);

                var formData = new FormData();
                formData.append('file', item.file);
                formData.append('training_id', TRAINING_ID);
                formData.append('threshold', threshRange.value);

                fetch('/api/predict/run', { method: 'POST', body: formData })
                    .then(function (resp) {
                        return resp.json().then(function (data) {
                            if (!resp.ok) throw new Error(data.error || 'Inference failed');
                            return data;
                        });
                    })
                    .then(function (data) {
                        item.annotatedImage = data.annotated_image;
                        item.detections = data.detections;
                        item.dataURI = null;
                        card.classList.remove('is-processing');
                        card.querySelector('img').src = data.annotated_image;

                        if (!keypointNames && data.detections) {
                            var firstWithKp = data.detections.find(function (d) { return d.keypoints && d.keypoints.length; });
                            if (firstWithKp) {
                                keypointNames = firstWithKp.keypoints.map(function (kp) { return kp.name; });
                                var frag = document.createDocumentFragment();
                                keypointNames.forEach(function (name) {
                                    var thX = document.createElement('th');
                                    thX.textContent = name + '-x';
                                    frag.appendChild(thX);
                                    var thY = document.createElement('th');
                                    thY.textContent = name + '-y';
                                    frag.appendChild(thY);
                                });
                                thead.appendChild(frag);
                            }
                        }

                        var rowFrag = document.createDocumentFragment();
                        (item.detections || []).forEach(function (det) {
                            globalRow++;
                            var row = document.createElement('tr');
                            row.dataset.imageIndex = idx;
                            var coords = det.bbox.map(function (v) { return Math.round(v); }).join(', ');
                            var html =
                                '<td>' + globalRow + '</td>' +
                                '<td title="' + escapeHtml(item.fileName) + '" style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeHtml(item.fileName) + '</td>' +
                                '<td>' + escapeHtml(det.class_name || 'unknown') + '</td>' +
                                '<td>' + (det.confidence ? Math.round(det.confidence * 100) + '%' : '\u2014') + '</td>' +
                                '<td style="font-family:monospace">' + coords + '</td>';
                            if (keypointNames) {
                                var kpMap = buildKeypointMap(det);
                                keypointNames.forEach(function (name) {
                                    var kp = kpMap[name];
                                    html += '<td style="font-family:monospace">' + (kp ? Math.round(kp.x) : '\u2014') + '</td>';
                                    html += '<td style="font-family:monospace">' + (kp ? Math.round(kp.y) : '\u2014') + '</td>';
                                });
                            }
                            row.innerHTML = html;
                            (function (capturedIdx) {
                                row.addEventListener('click', function () {
                                    highlightImage(capturedIdx);
                                    batchItems[capturedIdx].cardEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                                });
                            })(idx);
                            rowFrag.appendChild(row);
                        });
                        resultsTbody.appendChild(rowFrag);

                        completed++;
                        processNext();
                    })
                    .catch(function (e) {
                        card.classList.remove('is-processing');
                        card.style.opacity = '0.3';
                        nameEl.textContent = item.fileName + ' — error';
                        completed++;
                        processNext();
                    });
            }

            processNext();
        });

        // ---- CSV ----
        document.getElementById('download-csv-btn').addEventListener('click', function (e) {
            e.preventDefault();
            var header = '#,File,Class,Confidence,BBox';
            if (keypointNames) keypointNames.forEach(function (name) { header += ',' + name + '-x,' + name + '-y'; });
            var lines = [header];
            var c = 0;
            batchItems.forEach(function (item) {
                if (!item.detections) return;
                item.detections.forEach(function (det) {
                    c++;
                    var line = c + ',"' + item.fileName.replace(/"/g, '""') + '","' + (det.class_name || 'unknown') + '",' + (det.confidence ? (det.confidence * 100).toFixed(1) + '%' : '') + ',"' + det.bbox.map(function (v) { return Math.round(v); }).join(' ') + '"';
                    if (keypointNames) {
                        var kpMap = buildKeypointMap(det);
                        keypointNames.forEach(function (name) {
                            var kp = kpMap[name];
                            line += ',' + (kp ? Math.round(kp.x) : '') + ',' + (kp ? Math.round(kp.y) : '');
                        });
                    }
                    lines.push(line);
                });
            });
            var blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'detections.csv';
            a.click();
            URL.revokeObjectURL(url);
        });

    // ========================================================================
    // WORKFLOW MODE — single image (unchanged)
    // ========================================================================
    } else {
        var previewImg = document.getElementById('preview-img');
        var previewWrap = document.getElementById('preview-wrap');
        var resultsSection = document.getElementById('results-section');
        var resultsTbody = document.getElementById('results-tbody');
        var resultsSummary = document.getElementById('results-summary');
        var selectedFile = null;
        var selectedDataURI = null;

        dropZone.addEventListener('drop', function (e) {
            e.preventDefault();
            dropZone.classList.remove('is-dragover');
            var file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) loadFile(file);
        });
        fileInput.addEventListener('change', function () {
            if (fileInput.files[0]) loadFile(fileInput.files[0]);
        });

        function loadFile(file) {
            selectedFile = file;
            dropZoneText.textContent = file.name;
            var reader = new FileReader();
            reader.onload = function (e) {
                selectedDataURI = e.target.result;
                previewImg.src = selectedDataURI;
                emptyState.style.display = 'none';
                previewWrap.style.display = '';
                resultsSection.style.display = 'none';
                resultsTbody.innerHTML = '';
            };
            reader.readAsDataURL(file);
        }

        function showResults(items) {
            resultsTbody.innerHTML = '';
            items.forEach(function (item, i) {
                var row = document.createElement('tr');
                row.innerHTML =
                    '<td>' + (i + 1) + '</td>' +
                    '<td>' + (item.class_name || 'unknown') + '</td>' +
                    '<td>' + (item.confidence ? Math.round(item.confidence * 100) + '%' : '\u2014') + '</td>' +
                    '<td style="font-family:monospace">' + item.bbox.map(function (v) { return Math.round(v); }).join(', ') + '</td>';
                resultsTbody.appendChild(row);
            });
            resultsSummary.textContent = items.length + ' detection' + (items.length !== 1 ? 's' : '');
            resultsSection.style.display = '';
        }

        function runWorkflow() {
            var wf = JSON.parse(JSON.stringify(WORKFLOW_JSON));
            var nodes = wf.nodes || [];
            var mediaNode = nodes.find(function (n) { return n.data && (n.data.toolType === 'MediaInput' || n.data.nodeType === 'MediaInput'); });
            if (!mediaNode) throw new Error('This workflow has no image input (MediaInput node).');
            if (!mediaNode.data.parameters) mediaNode.data.parameters = {};
            mediaNode.data.parameters.data = selectedDataURI;

            return fetch('/agentui/api/workflow/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workflow: wf }),
            })
            .then(function (resp) { return resp.json(); })
            .then(function (data) {
                if (!data.success) throw new Error(data.error || 'Workflow execution failed');
                var outputImage = null, outputDetections = null;
                for (var nodeId in data.results) {
                    var result = data.results[nodeId];
                    if (!result.is_terminal) continue;
                    for (var key in result.outputs) {
                        var value = result.outputs[key];
                        if (typeof value === 'string' && value.startsWith('data:image/')) outputImage = value;
                        else if (Array.isArray(value) && value.length > 0 && value[0].bbox) outputDetections = value;
                    }
                }
                if (outputImage) previewImg.src = outputImage;
                if (outputDetections) showResults(outputDetections);
                if (!outputImage && !outputDetections) {
                    resultsSummary.textContent = 'Workflow produced no visual output';
                    resultsSection.style.display = '';
                }
            });
        }

        runBtn.addEventListener('click', function () {
            if (!selectedFile) { showError('Please drop an image first.'); return; }
            hideError();
            runBtn.classList.add('is-loading');
            runBtn.disabled = true;
            runWorkflow().catch(function (e) { showError(e.message); }).then(function () {
                runBtn.classList.remove('is-loading');
                runBtn.disabled = false;
            });
        });
    }
})();

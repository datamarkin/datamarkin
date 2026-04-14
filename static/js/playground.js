/**
 * Studio playground — image upload, model/workflow inference, results display.
 * Used by studio_playground.html.
 *
 * Expects PAGE_CONFIG: { mode, trainingId, workflowJson }
 */
(function () {
    var MODE = PAGE_CONFIG.mode;
    var TRAINING_ID = PAGE_CONFIG.trainingId;
    var WORKFLOW_JSON = PAGE_CONFIG.workflowJson;

    var dropZone = document.getElementById('drop-zone');
    var fileInput = document.getElementById('file-input');
    var previewImg = document.getElementById('preview-img');
    var previewWrap = document.getElementById('preview-wrap');
    var runBtn = document.getElementById('run-btn');
    var threshRange = document.getElementById('threshold-range');
    var threshDisp = document.getElementById('threshold-display');
    var errorMsg = document.getElementById('error-msg');
    var emptyState = document.getElementById('empty-state');
    var resultsSection = document.getElementById('results-section');
    var resultsTbody = document.getElementById('results-tbody');
    var resultsSummary = document.getElementById('results-summary');

    var selectedFile = null;
    var selectedDataURI = null;

    if (threshRange) {
        threshRange.addEventListener('input', function() {
            threshDisp.textContent = parseFloat(threshRange.value).toFixed(2);
        });
    }

    dropZone.addEventListener('click', function() { fileInput.click(); });
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('is-dragover');
    });
    dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('is-dragover'); });
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('is-dragover');
        var file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) loadFile(file);
    });
    fileInput.addEventListener('change', function() {
        if (fileInput.files[0]) loadFile(fileInput.files[0]);
    });

    function loadFile(file) {
        selectedFile = file;
        document.getElementById('drop-zone-text').textContent = file.name;
        var reader = new FileReader();
        reader.onload = function(e) {
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
        items.forEach(function(item, i) {
            var name = item.class_name || 'unknown';
            var coords = item.bbox.map(function(v) { return Math.round(v); }).join(', ');
            var row = document.createElement('tr');
            row.innerHTML =
                '<td>' + (i + 1) + '</td>' +
                '<td>' + name + '</td>' +
                '<td>' + (item.confidence ? Math.round(item.confidence * 100) + '%' : '—') + '</td>' +
                '<td style="font-family:monospace">' + coords + '</td>';
            resultsTbody.appendChild(row);
        });
        resultsSummary.textContent = items.length + ' detection' + (items.length !== 1 ? 's' : '');
        resultsSection.style.display = '';
    }

    function runModel() {
        var formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('training_id', TRAINING_ID);
        formData.append('threshold', threshRange.value);

        return fetch('/api/predict/run', {method: 'POST', body: formData})
            .then(function(resp) {
                return resp.json().then(function(data) {
                    if (!resp.ok) throw new Error(data.error || 'Inference failed');
                    previewImg.src = data.annotated_image;
                    showResults(data.detections);
                });
            });
    }

    function runWorkflow() {
        var wf = JSON.parse(JSON.stringify(WORKFLOW_JSON));
        var nodes = wf.nodes || [];
        var mediaNode = nodes.find(function(n) { return n.data && (n.data.toolType === 'MediaInput' || n.data.nodeType === 'MediaInput'); });
        if (!mediaNode) throw new Error('This workflow has no image input (MediaInput node).');
        if (!mediaNode.data.parameters) mediaNode.data.parameters = {};
        mediaNode.data.parameters.data = selectedDataURI;

        return fetch('/agentui/api/workflow/execute', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({workflow: wf}),
        })
        .then(function(resp) { return resp.json(); })
        .then(function(data) {
            if (!data.success) throw new Error(data.error || 'Workflow execution failed');

            var outputImage = null;
            var outputDetections = null;

            for (var nodeId in data.results) {
                var result = data.results[nodeId];
                if (!result.is_terminal) continue;
                for (var key in result.outputs) {
                    var value = result.outputs[key];
                    if (typeof value === 'string' && value.startsWith('data:image/')) {
                        outputImage = value;
                    } else if (Array.isArray(value) && value.length > 0 && value[0].bbox) {
                        outputDetections = value;
                    }
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

    runBtn.addEventListener('click', function() {
        if (!selectedFile) {
            showError('Please drop an image first.');
            return;
        }

        hideError();
        runBtn.classList.add('is-loading');
        runBtn.disabled = true;

        var p = MODE === 'model' ? runModel() : runWorkflow();
        p.catch(function(e) {
            showError(e.message);
        }).then(function() {
            runBtn.classList.remove('is-loading');
            runBtn.disabled = false;
        });
    });

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('is-hidden');
    }

    function hideError() {
        errorMsg.textContent = '';
        errorMsg.classList.add('is-hidden');
    }
})();

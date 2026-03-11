(function () {
    var cfg = window.STUDIO_CONFIG;
    var labels = cfg.labels;
    var existingAnnotations = cfg.annotations;
    var imageWidth = cfg.imageWidth;
    var imageHeight = cfg.imageHeight;
    var files = cfg.files;
    var currentIndex = cfg.currentIndex;
    var projectId = cfg.projectId;
    var fileId = cfg.fileId;
    var embeddingId = cfg.embeddingId;
    var samReady = embeddingId !== null;
    var samActive = false;

    // Build label lookup map: id -> {name, color, keypoints}
    var labelMap = {};
    labels.forEach(function (l) { labelMap[l.id] = l; });

    // Build reverse map: label name -> id (for export conversion)
    var labelNameToId = {};
    labels.forEach(function (l) { labelNameToId[l.name] = l.id; });

    // -------------------------------------------------------------------------
    // Color helpers
    // -------------------------------------------------------------------------
    function hexToFill(hex) {
        var r = parseInt(hex.substring(0, 2), 16);
        var g = parseInt(hex.substring(2, 4), 16);
        var b = parseInt(hex.substring(4, 6), 16);
        return 'rgba(' + r + ',' + g + ',' + b + ',0.2)';
    }

    // -------------------------------------------------------------------------
    // Coordinate converters (normalized 0-1 -> pixel)
    // -------------------------------------------------------------------------
    function dbBboxToPixelBbox(b) {
        return [b[0] * imageWidth, b[1] * imageHeight,
                (b[0] + b[2]) * imageWidth, (b[1] + b[3]) * imageHeight];
    }

    function dbSegmentationToPixel(seg) {
        var out = [];
        for (var i = 0; i < seg.length; i += 2) {
            out.push(seg[i] * imageWidth, seg[i + 1] * imageHeight);
        }
        return out;
    }

    function dbKeypointsToPixel(kps, labelObj) {
        var kpDefs = (labelObj && labelObj.keypoints) || [];
        return kps.map(function (kp) {
            var def = kpDefs.find(function (d) { return d.id === kp.id; });
            return {
                name: def ? def.name : 'kp-' + kp.id,
                point: [kp.point[0] * imageWidth, kp.point[1] * imageHeight]
            };
        });
    }

    // -------------------------------------------------------------------------
    // Convert MarkinJS export format -> DB format
    // MarkinJS: {annotations: [{class: "name", bbox: {x,y,width,height}, segmentation: [...]}]}
    // DB:       {objects: [{class: id, bbox: [x,y,w,h], segmentation: [...]}]}
    // -------------------------------------------------------------------------
    function exportToDbFormat(exported) {
        if (!exported || !exported.annotations) return null;
        var objects = exported.annotations.map(function (ann) {
            var obj = {};

            // class: resolve name back to id
            var classVal = ann.class;
            if (typeof classVal === 'string' && classVal !== '') {
                var asInt = parseInt(classVal, 10);
                if (!isNaN(asInt) && String(asInt) === classVal) {
                    obj.class = asInt;
                } else {
                    obj.class = (labelNameToId[classVal] !== undefined) ? labelNameToId[classVal] : 0;
                }
            } else {
                obj.class = typeof classVal === 'number' ? classVal : 0;
            }

            if (ann.bbox) {
                obj.bbox = [ann.bbox.x, ann.bbox.y, ann.bbox.width, ann.bbox.height];
            }
            if (ann.segmentation && ann.segmentation.length >= 6) {
                obj.segmentation = ann.segmentation;
            }
            if (ann.keypoints && ann.keypoints.length > 0) {
                obj.keypoints = ann.keypoints.map(function (kp) {
                    var labelObj = labelMap[obj.class] || {};
                    var kpDefs = labelObj.keypoints || [];
                    var def = kpDefs.find(function (d) { return d.name === kp.name; });
                    return {id: def ? def.id : 0, point: kp.point};
                });
            }
            return obj;
        });
        return {objects: objects};
    }

    // -------------------------------------------------------------------------
    // Save annotations to API
    // -------------------------------------------------------------------------
    function saveAnnotations() {
        if (!window._annotator) return;
        var exported = window._annotator.exportAllAnnotations({
            normalize: true,
            width: imageWidth,
            height: imageHeight
        });
        var dbData = exportToDbFormat(exported);
        fetch('/api/files/' + fileId, {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({annotations: dbData})
        });
    }

    // -------------------------------------------------------------------------
    // Apply SAM masks as new annotations
    // -------------------------------------------------------------------------
    function applyMasks(masks) {
        if (!window._annotator) return;
        var activeBtn = document.querySelector('.sidebar-label-btn.is-active');
        var labelId = activeBtn ? parseInt(activeBtn.dataset.label) : 0;
        var label = labelMap[labelId] || {};
        var color = label.color || 'ff0000';

        masks.forEach(function (mask) {
            if (!mask.segmentation || !mask.segmentation.length) return;
            var pixelSeg = [];
            for (var i = 0; i < mask.segmentation.length; i += 2) {
                pixelSeg.push(mask.segmentation[i] * imageWidth,
                               mask.segmentation[i + 1] * imageHeight);
            }
            var bbox = mask.bbox ? [
                mask.bbox[0] * imageWidth, mask.bbox[1] * imageHeight,
                (mask.bbox[0] + mask.bbox[2]) * imageWidth,
                (mask.bbox[1] + mask.bbox[3]) * imageHeight
            ] : null;
            window._annotator.createAnnotation({
                class: String(labelId),
                stroke: '#' + color,
                fill: hexToFill(color),
                segmentation: pixelSeg,
                bbox: bbox
            });
        });

        if (masks.length > 0) {
            window._annotator.deselect();
            saveAnnotations();
        }
    }

    // -------------------------------------------------------------------------
    // Initialise MarkinJS annotator (called after image loads)
    // -------------------------------------------------------------------------
    function initAnnotator() {
        var annotator = MarkinJS.createImageAnnotator('annotate-image', {});
        window._annotator = annotator;

        // Load existing annotations from server
        if (existingAnnotations && existingAnnotations.objects) {
            existingAnnotations.objects.forEach(function (obj) {
                var label = labelMap[obj.class] || {};
                var color = label.color || 'ff0000';
                var opts = {
                    class: String(obj.class),
                    stroke: '#' + color,
                    fill: hexToFill(color)
                };
                if (obj.bbox) opts.bbox = dbBboxToPixelBbox(obj.bbox);
                if (obj.segmentation) opts.segmentation = dbSegmentationToPixel(obj.segmentation);
                if (obj.keypoints) opts.keypoints = dbKeypointsToPixel(obj.keypoints, label);
                annotator.createAnnotation(opts);
            });
            annotator.deselect();
        }

        annotator.clearHistory();

        // Auto-save on any annotation change
        annotator.on('annotationmodificationcomplete', saveAnnotations);
        annotator.on('annotationcreated', saveAnnotations);

        // ---- SAM click handler (on SVG overlay, NOT the image element) ----
        if (samReady) {
            var svgEl = annotator.getSVGElement();

            svgEl.addEventListener('click', function (e) {
                if (!samActive) return;
                // getBoundingClientRect gives actual rendered bounds (accounts for Zoomist scale/pan)
                var rect = svgEl.getBoundingClientRect();
                var x = (e.clientX - rect.left) / rect.width;
                var y = (e.clientY - rect.top) / rect.height;
                var polarity = e.shiftKey ? -1 : 1;

                fetch('/api/sam/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        embedding_id: embeddingId,
                        prompts: [{type: 'point', x: x, y: y, polarity: polarity}],
                        width: imageWidth,
                        height: imageHeight
                    })
                })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.data && data.data.masks) applyMasks(data.data.masks);
                });
            });
        }
    }

    // -------------------------------------------------------------------------
    // Wait for image load before initialising
    // -------------------------------------------------------------------------
    var img = document.getElementById('annotate-image');
    if (img.complete) {
        initAnnotator();
    } else {
        img.addEventListener('load', initAnnotator);
    }

    // -------------------------------------------------------------------------
    // Label click handler (event delegation)
    // -------------------------------------------------------------------------
    document.getElementById('label-list').addEventListener('click', function (e) {
        var btn = e.target.closest('.sidebar-label-btn');
        if (!btn) return;
        this.querySelectorAll('.sidebar-label-btn').forEach(function (b) {
            b.classList.remove('is-active');
        });
        btn.classList.add('is-active');
    });

    // -------------------------------------------------------------------------
    // SAM toggle button
    // -------------------------------------------------------------------------
    if (samReady) {
        var samToggleBtn = document.getElementById('sam-toggle-btn');
        samToggleBtn.addEventListener('click', function () {
            samActive = !samActive;
            samToggleBtn.classList.toggle('is-active', samActive);
            if (samActive) {
                // Disable MarkinJS draw/select while SAM is active
                if (window._annotator) window._annotator.disable();
            } else {
                if (window._annotator) window._annotator.enable();
            }
        });

        // SAM text prompt
        var textBtn = document.getElementById('sam-text-btn');
        if (textBtn) {
            textBtn.addEventListener('click', function () {
                var prompt = document.getElementById('sam-text-input').value.trim();
                if (!prompt) return;
                textBtn.classList.add('is-loading');
                fetch('/api/sam/text', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        embedding_id: embeddingId,
                        text_prompt: prompt,
                        width: imageWidth,
                        height: imageHeight,
                        confidence_threshold: 0.1
                    })
                })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    textBtn.classList.remove('is-loading');
                    if (data.data && data.data.masks) applyMasks(data.data.masks);
                })
                .catch(function () { textBtn.classList.remove('is-loading'); });
            });
        }
    }

    // -------------------------------------------------------------------------
    // Arrow key navigation
    // -------------------------------------------------------------------------
    document.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        if (window._annotator && window._annotator.getSelectedElement()) return;

        if (e.key === 'ArrowLeft' && currentIndex > 0) {
            window.location.href = '/project/' + projectId + '/' + files[currentIndex - 1].id;
        } else if (e.key === 'ArrowRight' && currentIndex < files.length - 1) {
            window.location.href = '/project/' + projectId + '/' + files[currentIndex + 1].id;
        }
    });
})();

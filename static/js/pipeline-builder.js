/**
 * Pipeline Builder — pre-processing and augmentation pipeline UI
 * Used by the project settings modal in project.html
 */

var _pipelineSaveTimer = null;

const PP_CATALOG = [
    { category: 'Geometric', items: [
        { type: 'Resize', label: 'Resize', params: [
            { name: 'width', label: 'Width', default: 640, min: 1 },
            { name: 'height', label: 'Height', default: 640, min: 1 },
        ]},
        { type: 'CenterCrop', label: 'Center Crop', params: [
            { name: 'width', label: 'Width', default: 512, min: 1 },
            { name: 'height', label: 'Height', default: 512, min: 1 },
        ]},
        { type: 'PadIfNeeded', label: 'Pad If Needed', params: [
            { name: 'min_height', label: 'Min Height', default: 640, min: 1 },
            { name: 'min_width', label: 'Min Width', default: 640, min: 1 },
        ]},
    ]},
    { category: 'Color', items: [
        { type: 'Normalize', label: 'Normalize', params: [
            { name: 'mean_r', label: 'Mean R', default: 0.485, min: 0, max: 1, step: 0.001 },
            { name: 'mean_g', label: 'Mean G', default: 0.456, min: 0, max: 1, step: 0.001 },
            { name: 'mean_b', label: 'Mean B', default: 0.406, min: 0, max: 1, step: 0.001 },
            { name: 'std_r', label: 'Std R', default: 0.229, min: 0, max: 1, step: 0.001 },
            { name: 'std_g', label: 'Std G', default: 0.224, min: 0, max: 1, step: 0.001 },
            { name: 'std_b', label: 'Std B', default: 0.225, min: 0, max: 1, step: 0.001 },
        ]},
        { type: 'ToGray', label: 'To Grayscale', params: [] },
        { type: 'CLAHE', label: 'CLAHE', params: [
            { name: 'clip_limit', label: 'Clip Limit', default: 2.0, min: 0.1, step: 0.1 },
        ]},
    ]},
];

const AUG_CATALOG = [
    { category: 'Geometric', items: [
        { type: 'HorizontalFlip', label: 'Horizontal Flip', params: [] },
        { type: 'VerticalFlip', label: 'Vertical Flip', params: [] },
        { type: 'RandomRotate90', label: 'Random Rotate 90°', params: [] },
        { type: 'Rotate', label: 'Rotate', params: [
            { name: 'limit', label: 'Angle Limit (°)', default: 45, min: 0, max: 180 },
        ]},
        { type: 'ShiftScaleRotate', label: 'Shift Scale Rotate', params: [
            { name: 'shift_limit', label: 'Shift', default: 0.1, min: 0, max: 0.5, step: 0.01 },
            { name: 'scale_limit', label: 'Scale', default: 0.1, min: 0, max: 0.5, step: 0.01 },
            { name: 'rotate_limit', label: 'Rotate (°)', default: 45, min: 0, max: 180 },
        ]},
        { type: 'Perspective', label: 'Perspective', params: [
            { name: 'scale', label: 'Scale', default: 0.05, min: 0.01, max: 0.3, step: 0.01 },
        ]},
        { type: 'ElasticTransform', label: 'Elastic Transform', params: [
            { name: 'alpha', label: 'Alpha', default: 1, min: 0, step: 0.1 },
            { name: 'sigma', label: 'Sigma', default: 50, min: 1, step: 1 },
        ]},
    ]},
    { category: 'Color', items: [
        { type: 'ColorJitter', label: 'Color Jitter', params: [
            { name: 'brightness', label: 'Brightness', default: 0.2, min: 0, max: 1, step: 0.01 },
            { name: 'contrast', label: 'Contrast', default: 0.2, min: 0, max: 1, step: 0.01 },
            { name: 'saturation', label: 'Saturation', default: 0.2, min: 0, max: 1, step: 0.01 },
            { name: 'hue', label: 'Hue', default: 0.1, min: 0, max: 0.5, step: 0.01 },
        ]},
        { type: 'HueSaturationValue', label: 'Hue Sat Value', params: [
            { name: 'hue_shift_limit', label: 'Hue Shift', default: 20, min: 0, max: 180 },
            { name: 'sat_shift_limit', label: 'Sat Shift', default: 30, min: 0, max: 255 },
            { name: 'val_shift_limit', label: 'Val Shift', default: 20, min: 0, max: 255 },
        ]},
        { type: 'RGBShift', label: 'RGB Shift', params: [
            { name: 'r_shift_limit', label: 'R Shift', default: 15, min: 0, max: 255 },
            { name: 'g_shift_limit', label: 'G Shift', default: 15, min: 0, max: 255 },
            { name: 'b_shift_limit', label: 'B Shift', default: 15, min: 0, max: 255 },
        ]},
        { type: 'ChannelShuffle', label: 'Channel Shuffle', params: [] },
    ]},
    { category: 'Blur / Noise', items: [
        { type: 'GaussianBlur', label: 'Gaussian Blur', params: [
            { name: 'blur_limit', label: 'Blur Limit', default: 7, min: 3, step: 2 },
        ]},
        { type: 'MotionBlur', label: 'Motion Blur', params: [
            { name: 'blur_limit', label: 'Blur Limit', default: 7, min: 3, step: 2 },
        ]},
        { type: 'GaussNoise', label: 'Gaussian Noise', params: [
            { name: 'var_limit', label: 'Variance', default: 10, min: 0, step: 0.5 },
        ]},
        { type: 'ISONoise', label: 'ISO Noise', params: [
            { name: 'color_shift', label: 'Color Shift', default: 0.05, min: 0, max: 1, step: 0.01 },
            { name: 'intensity', label: 'Intensity', default: 0.5, min: 0, max: 1, step: 0.01 },
        ]},
        { type: 'JpegCompression', label: 'JPEG Compression', params: [
            { name: 'quality_lower', label: 'Quality Min', default: 50, min: 0, max: 100 },
            { name: 'quality_upper', label: 'Quality Max', default: 99, min: 0, max: 100 },
        ]},
    ]},
    { category: 'Composite', items: [
        { type: 'OneOf', label: 'OneOf Group', params: [] },
    ]},
];

function findDef(type, catalog) {
    for (const group of catalog) {
        for (const item of group.items) {
            if (item.type === type) return item;
        }
    }
    return null;
}

function buildCatalog(containerEl, catalog) {
    catalog.forEach(function(group) {
        const details = document.createElement('details');
        details.open = true;
        const summary = document.createElement('summary');
        summary.className = 'catalog-category';
        summary.textContent = group.category;
        details.appendChild(summary);
        const ul = document.createElement('ul');
        ul.className = 'catalog-items';
        group.items.forEach(function(item) {
            const li = document.createElement('li');
            li.className = 'catalog-item';
            li.dataset.type = item.type;
            li.textContent = item.label;
            ul.appendChild(li);
        });
        details.appendChild(ul);
        containerEl.appendChild(details);
    });
}

function createTransformCard(def, isAug) {
    const li = document.createElement('li');
    li.className = 'transform-card' + (def.type === 'OneOf' ? ' oneof-card' : '');
    li.dataset.type = def.type;

    const header = document.createElement('div');
    header.className = 'transform-card-header';

    const handle = document.createElement('span');
    handle.className = 'drag-handle';
    handle.textContent = '⠿';
    header.appendChild(handle);

    const labelEl = document.createElement('span');
    labelEl.className = 'transform-label';
    labelEl.textContent = def.label;
    header.appendChild(labelEl);

    if (isAug) {
        const pCtrl = document.createElement('div');
        pCtrl.className = 'p-control';
        const pLbl = document.createElement('span');
        pLbl.className = 'p-label';
        pLbl.textContent = 'p=';
        const pSlider = document.createElement('input');
        pSlider.type = 'range';
        pSlider.min = 0; pSlider.max = 1; pSlider.step = 0.05; pSlider.value = 0.5;
        pSlider.className = 'p-slider';
        const pVal = document.createElement('span');
        pVal.className = 'p-value';
        pVal.textContent = '0.50';
        pSlider.addEventListener('input', function() {
            pVal.textContent = parseFloat(this.value).toFixed(2);
            var root = li.closest('.pipeline-stage-col');
            if (root) schedulePipelineSave(root.querySelector('.pipeline-list'), true);
        });
        pCtrl.appendChild(pLbl);
        pCtrl.appendChild(pSlider);
        pCtrl.appendChild(pVal);
        header.appendChild(pCtrl);
    }

    const del = document.createElement('button');
    del.className = 'transform-delete';
    del.textContent = '×';
    del.addEventListener('click', function(e) {
        e.stopPropagation();
        var root = li.closest('.pipeline-stage-col');
        var isAugPanel = li.closest('#panel-augmentation') !== null;
        li.remove();
        if (root) schedulePipelineSave(root.querySelector('.pipeline-list'), isAugPanel);
    });
    header.appendChild(del);
    li.appendChild(header);

    if (def.type === 'OneOf') {
        const inner = document.createElement('div');
        inner.className = 'oneof-inner';
        const hint = document.createElement('p');
        hint.className = 'oneof-hint';
        hint.textContent = 'Drop transforms here — one is picked randomly';
        inner.appendChild(hint);
        const innerList = document.createElement('ul');
        innerList.className = 'pipeline-list oneof-list';
        inner.appendChild(innerList);
        li.appendChild(inner);
        setTimeout(function() { initPipelineSortable(innerList, AUG_CATALOG, true); }, 0);
    } else if (def.params.length > 0) {
        const paramsEl = document.createElement('div');
        paramsEl.className = 'transform-card-params';
        paramsEl.style.display = 'none';
        def.params.forEach(function(param) {
            const row = document.createElement('div');
            row.className = 'param-row';
            const lbl = document.createElement('label');
            lbl.textContent = param.label;
            row.appendChild(lbl);
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.value = param.default;
            inp.dataset.param = param.name;
            if (param.min !== undefined) inp.min = param.min;
            if (param.max !== undefined) inp.max = param.max;
            if (param.step !== undefined) inp.step = param.step;
            inp.addEventListener('change', function() {
                var root = li.closest('.pipeline-stage-col');
                var isAugPanel = li.closest('#panel-augmentation') !== null;
                if (root) schedulePipelineSave(root.querySelector('.pipeline-list'), isAugPanel);
            });
            row.appendChild(inp);
            paramsEl.appendChild(row);
        });
        li.appendChild(paramsEl);
        header.style.cursor = 'pointer';
        header.addEventListener('click', function(e) {
            if (del.contains(e.target)) return;
            paramsEl.style.display = paramsEl.style.display === 'none' ? 'block' : 'none';
        });
    }
    return li;
}

function initPipelineSortable(pipelineEl, catalog, isAug) {
    if (typeof Sortable === 'undefined') return;
    Sortable.create(pipelineEl, {
        group: isAug ? 'aug-pipeline' : 'pp-pipeline',
        animation: 150,
        handle: '.drag-handle',
        onAdd: function(evt) {
            const type = evt.item.dataset.type;
            const def = findDef(type, catalog);
            if (!def) { evt.item.remove(); return; }
            const card = createTransformCard(def, isAug);
            pipelineEl.insertBefore(card, evt.item);
            evt.item.remove();
            schedulePipelineSave(pipelineEl, isAug);
        },
        onUpdate: function() { schedulePipelineSave(pipelineEl, isAug); },
    });
}

function initCatalogSortable(catalogEl, isAug) {
    if (typeof Sortable === 'undefined') return;
    const groupName = isAug ? 'aug-pipeline' : 'pp-pipeline';
    catalogEl.querySelectorAll('.catalog-items').forEach(function(ul) {
        Sortable.create(ul, {
            group: { name: groupName, pull: 'clone', put: false },
            sort: false,
            animation: 150,
        });
    });
}

function initCatalogClicks(catalogEl, pipelineEl, catalog, isAug) {
    catalogEl.addEventListener('click', function(e) {
        const item = e.target.closest('.catalog-item');
        if (!item) return;
        const def = findDef(item.dataset.type, catalog);
        if (!def) return;
        pipelineEl.appendChild(createTransformCard(def, isAug));
        schedulePipelineSave(pipelineEl, isAug);
    });
}

function serializePipeline(pipelineEl, isAug) {
    const transforms = [];
    pipelineEl.querySelectorAll(':scope > .transform-card').forEach(function(card) {
        const type = card.dataset.type;
        const t = { type: type };
        if (isAug) {
            const sl = card.querySelector('.p-slider');
            t.p = sl ? parseFloat(sl.value) : 1.0;
        }
        if (type === 'OneOf') {
            const inner = card.querySelector('.oneof-list');
            t.children = inner ? serializeChildren(inner, isAug) : [];
        } else {
            t.params = {};
            card.querySelectorAll('.param-row input[data-param]').forEach(function(inp) {
                t.params[inp.dataset.param] = parseFloat(inp.value);
            });
        }
        transforms.push(t);
    });
    return { transforms: transforms };
}

function serializeChildren(listEl, isAug) {
    const out = [];
    listEl.querySelectorAll(':scope > .transform-card').forEach(function(card) {
        const t = { type: card.dataset.type, params: {} };
        if (isAug) {
            const sl = card.querySelector('.p-slider');
            t.p = sl ? parseFloat(sl.value) : 1.0;
        }
        card.querySelectorAll('.param-row input[data-param]').forEach(function(inp) {
            t.params[inp.dataset.param] = parseFloat(inp.value);
        });
        out.push(t);
    });
    return out;
}

function schedulePipelineSave(pipelineEl, isAug) {
    // Always serialize from the root pipeline, not a nested oneof-list
    var rootEl = document.getElementById(isAug ? 'aug-pipeline' : 'pp-pipeline') || pipelineEl;
    clearTimeout(_pipelineSaveTimer);
    _pipelineSaveTimer = setTimeout(function() {
        var key = isAug ? 'augmentation' : 'preprocessing';
        fetch('/project/' + PAGE_CONFIG.projectId + '/pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: key, pipeline: serializePipeline(rootEl, isAug) }),
        });
    }, 600);
}

function loadPipeline(pipelineEl, pipelineJson, catalog, isAug) {
    if (!pipelineJson || !pipelineJson.transforms || !pipelineJson.transforms.length) return;
    pipelineJson.transforms.forEach(function(t) {
        const def = findDef(t.type, catalog);
        if (!def) return;
        const card = createTransformCard(def, isAug);
        if (t.params) {
            card.querySelectorAll('.param-row input[data-param]').forEach(function(inp) {
                if (t.params[inp.dataset.param] !== undefined) inp.value = t.params[inp.dataset.param];
            });
        }
        if (isAug && t.p !== undefined) {
            const sl = card.querySelector('.p-slider');
            const pv = card.querySelector('.p-value');
            if (sl) { sl.value = t.p; pv.textContent = parseFloat(t.p).toFixed(2); }
        }
        pipelineEl.appendChild(card);
        if (t.type === 'OneOf' && t.children && t.children.length) {
            setTimeout(function() {
                const innerList = card.querySelector('.oneof-list');
                if (!innerList) return;
                t.children.forEach(function(child) {
                    const childDef = findDef(child.type, catalog);
                    if (!childDef) return;
                    const cc = createTransformCard(childDef, isAug);
                    if (child.params) {
                        cc.querySelectorAll('.param-row input[data-param]').forEach(function(inp) {
                            if (child.params[inp.dataset.param] !== undefined) inp.value = child.params[inp.dataset.param];
                        });
                    }
                    if (isAug && child.p !== undefined) {
                        const sl = cc.querySelector('.p-slider');
                        const pv = cc.querySelector('.p-value');
                        if (sl) { sl.value = child.p; pv.textContent = parseFloat(child.p).toFixed(2); }
                    }
                    innerList.appendChild(cc);
                });
            }, 0);
        }
    });
}

function initSearch(searchEl, catalogEl) {
    searchEl.addEventListener('input', function() {
        const q = this.value.toLowerCase().trim();
        catalogEl.querySelectorAll('.catalog-item').forEach(function(item) {
            item.style.display = !q || item.textContent.toLowerCase().includes(q) ? '' : 'none';
        });
        catalogEl.querySelectorAll('details').forEach(function(d) {
            const any = Array.from(d.querySelectorAll('.catalog-item')).some(function(i) {
                return i.style.display !== 'none';
            });
            d.style.display = any ? '' : 'none';
        });
    });
}

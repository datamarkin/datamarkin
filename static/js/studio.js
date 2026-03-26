/**
 * DataMarkin Studio - Annotation Management
 * Manages annotations, interactions, and keypoint placement
 */

// Project configuration — projectId, projectType, projectLabels are defined inline by the template

const imageContainer = document.getElementById('image-to-annotate');

const imageId = imageContainer.dataset.imageId;
const imageWidth = parseInt(imageContainer.dataset.width);
const imageHeight = parseInt(imageContainer.dataset.height);
const imageRatio = parseInt(imageContainer.dataset.ratio);
const imageShape = [imageWidth, imageHeight];

// Opacity settings
const bboxFillOpacity = document.querySelector('html').dataset.bboxFillOpacity;
const bboxStrokeOpacity = document.querySelector('html').dataset.bboxStrokeOpacity;
const polygonFillOpacity = document.querySelector('html').dataset.polygonFillOpacity;
const polygonStrokeOpacity = document.querySelector('html').dataset.polygonStrokeOpacity;
const keypointStrokeOpacity = document.querySelector('html').dataset.keypointStrokeOpacity;
const keypointFillOpacity = document.querySelector('html').dataset.keypointFillOpacity;

const zoomistContainer = document.getElementById('zoomist-container');
const zoomistContainerWidth = zoomistContainer.getBoundingClientRect().width;
const zoomistContainerHeight = zoomistContainer.getBoundingClientRect().height;

// Handle drag vs. click conflict between Zoomist and VivaSVG
const MIN_DRAG_DISTANCE = 5; // pixels


let app = {
    previousBbox: null,
    mouseX: 0,
    mouseY: 0,
    svgX: 0,
    svgY: 0,
    imgX: 0,
    imgY: 0,
    isDragging: false,
    dragStartX: null,
    dragStartY: null,
    zoomistScale: 1,
    scaleX: 1,
    scaleY: 1,
    simplificationTolerance: 2.0,  // Polygon simplification
    samEnabled: true,               // SAM mode enabled
}


// Annotation state
const annotation = {
    selectedId: null,        // Currently selected annotation
    activeKeypointType: null, // Currently active keypoint for placement
    firstInitialised: false,      // Whether annotations initialized
    list: [],                 // List of annotations
    bbox: [0, 0, imageWidth, imageHeight],
    accumulatedLabels: [1],
    accumulatedPoints: [],
    croppedBbox: [0, 0, imageWidth, imageHeight],
    hasLowMask: false,
    activeKeypointType: null
};

// SVG element reference
const svgElement = markin.getSVGElement();

// Capture mousedown to record starting position
svgElement.addEventListener('mousedown', function (e) {
    app.dragStartX = e.clientX;
    app.dragStartY = e.clientY;
    app.isDragging = false;
}, true);

// Add click handler to SVG for placing keypoints
svgElement.addEventListener('click', function (e) {
    // Only proceed if a keypoint type is selected
    if (!annotation.activeKeypointType) return;

    try {
        const selected = markin.getSelectedElement();
        if (!selected) {
            console.log('Error: No annotation selected');
            return;
        }

        // Add the keypoint at the transformed position
        const keypoint = markin.addKeypoint(null, annotation.activeKeypointType, app.svgX, app.svgY);
        if (keypoint) {
            console.log(`Added keypoint "${annotation.activeKeypointType}" at SVG coordinates (${app.svgX}, ${app.svgY})`);
        } else {
            console.log('Failed to add keypoint');
        }
    } catch (error) {
        console.error('Add keypoint error:', error);
    }
});

// ── SAM model download ────────────────────────────────────────────────────────

let _samModelReady = false;
let _samDownloadPollTimer = null;

function _showDownloadBanner(pct) {
    const banner = document.getElementById('sam-download-banner');
    if (banner) {
        banner.style.display = '';
        document.getElementById('sam-dl-pct').textContent = pct + '%';
        document.getElementById('sam-dl-progress').value = pct;
    }
}

function _hideDownloadBanner() {
    const banner = document.getElementById('sam-download-banner');
    if (banner) banner.style.display = 'none';
}

function _pollModelDownload() {
    if (_samDownloadPollTimer) clearInterval(_samDownloadPollTimer);
    _samDownloadPollTimer = setInterval(() => {
        fetch('/api/sam/model_status')
            .then(r => r.json())
            .then(resp => {
                const s = resp.data;
                if (s.status === 'ready') {
                    clearInterval(_samDownloadPollTimer);
                    _samModelReady = true;
                    _hideDownloadBanner();
                } else if (s.status === 'error') {
                    clearInterval(_samDownloadPollTimer);
                    _hideDownloadBanner();
                    console.error('SAM model download failed:', s.error);
                } else {
                    _showDownloadBanner(s.pct || 0);
                }
            })
            .catch(() => {});
    }, 1000);
}

async function _ensureSAMModel() {
    const resp = await fetch('/api/sam/model_status').then(r => r.json());
    const s = resp.data;
    if (s.status === 'ready') {
        _samModelReady = true;
        return;
    }
    // Trigger download (idempotent if already in progress)
    await fetch('/api/sam/download_model', { method: 'POST' });
    _showDownloadBanner(0);
    _pollModelDownload();
}

// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Check SAM model availability; auto-download if needed
    if (projectType !== 'keypoint-detection' && projectType !== 'image-classification') {
        _ensureSAMModel();
    }

    // Setup keypoint buttons
    setupKeypointButtons();

    // Disable SAM for keypoint-detection (markin handles bbox drawing directly)
    if (projectType === 'keypoint-detection') {
        toggleSAM();
    }

    zoomist.on('zoom', () => {
        app.zoomistScale = zoomist.transform.scale;
        updateGuidelines()
        updateCroppedBbox()

        markin.setZoom(app.zoomistScale)
    });

    markin.on('annotationmodificationcomplete', function(data) {
        markin.saveState(`${data.modificationType}_complete`);
        const exported = markin.exportAllAnnotations({
            normalize: true,
            width: imageWidth,
            height: imageHeight
        });
        updateAPIAnnotations(normalizeAnnotationsForStorage(exported));
    });

    // Track mouse position on the SVG
    svgElement.addEventListener('mousemove', function (e) {
        // Update mouse positions
        updateMousePosition(e);

        // Update the cropped bbox
        updateCroppedBbox()

        // Update guidelines if they exist
        updateGuidelines();

        // Update zoomist if needs to be disabled
        updateZoomist()
    });

    // Add click event listeners to all label buttons
    document.querySelectorAll('.is-label-button').forEach(button => {
        button.addEventListener('click', handleLabelButtonClick);
    });

    svgElement.addEventListener('click', (event) => {
        if (!app.isDragging && app.samEnabled) {
            handleSamPointClick();
        }
    });

    // Add keyboard shortcut event listeners
    document.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter') {
            if (annotation.firstInitialised) {
                validateAnnotation();
                console.log("Annotation enter triggered");
            }
        }
    });

    document.getElementById('samToggle').addEventListener('click', async () => {
        toggleSAM();
        // Remove focus from the button after clicking
        document.getElementById('samToggle').blur();
        console.log("Annotation toggle triggered");
    });

    // Add keyboard shortcut event listeners
    document.addEventListener('keydown', async (event) => {
        if (event.key === 'd') {
            toggleSAM();
        }
    });

    document.getElementById('reset-annotations').addEventListener('click', async () => {
        document.getElementById('reset-annotations').blur();
        resetAnnotation();
        console.log("Annotation reset triggered");
    });

    document.getElementById('validate-annotations').addEventListener('click', async () => {
        if (annotation.firstInitialised && app.samEnabled) {
            document.getElementById('validate-annotations').blur();
            validateAnnotation();
        }
        console.log("Annotation validate triggered");
    });
})

function updateZoomist() {
    if (annotation.firstInitialised) {
        // Disable interactions
        zoomistContainer.classList.add('zoomist-not-draggable');
        zoomistContainer.classList.add('zoomist-not-wheelable');
    } else {
        // Re-enable interactions
        zoomistContainer.classList.remove('zoomist-not-draggable');
        zoomistContainer.classList.remove('zoomist-not-wheelable');
    }
}

function checkMouseOnSVG(e) {
    // Get the zoomistContainer element's boundaries
    // Don't get SVG because it can be scaled. 
    const svgRect = zoomistContainer.getBoundingClientRect();

    // We need client coordinates for comparison with element boundaries
    const clientX = e.clientX;
    const clientY = e.clientY;

    // Check if mouse is within the SVG boundaries
    if (clientX < svgRect.left || clientX > svgRect.right ||
        clientY < svgRect.top || clientY > svgRect.bottom) {
        if (!annotation.firstInitialised) {
            console.log("Deleting annotation canditate")
            const to_delete = document.getElementById('candidate-annotation')
            if (to_delete) {
                to_delete.remove()
            }
        }
        return false;
    }
    return true;
}

function setupKeypointButtons() {
    const keypointButtons = document.querySelectorAll('.is-keypoint-button');

    if (keypointButtons.length > 0) {
        // First deactivate all buttons to ensure clean UI state
        const deactivateAllButtons = () => {
            keypointButtons.forEach(btn => {
                btn.classList.remove('is-active');
            });
        };

        keypointButtons.forEach(button => {
            button.addEventListener('click', function () {
                const keypointName = this.dataset.keypointName;

                if (annotation.activeKeypointType === keypointName) {
                    // If clicking the active button, deactivate it
                    annotation.activeKeypointType = null;
                    deactivateAllButtons();
                    console.log('Keypoint placement cancelled');
                } else {
                    // Activate this button, deactivate others
                    deactivateAllButtons();
                    this.classList.add('is-active');
                    annotation.activeKeypointType = keypointName;
                    console.log(`Ready to place keypoint: ${keypointName}`);
                }
            });
        });
    }

    // Add global handler for custom "add-keypoint" button if it exists
    const addKeypointButton = document.getElementById('add-keypoint');
    if (addKeypointButton) {
        addKeypointButton.addEventListener('click', function () {
            if (annotation.activeKeypointType === "new-point") {
                // If already active, deactivate
                annotation.activeKeypointType = null;
                this.classList.remove('is-active');
            } else {
                // Deactivate other buttons and activate this one
                const keypointButtons = document.querySelectorAll('.is-keypoint-button');
                keypointButtons.forEach(btn => {
                    btn.classList.remove('is-active');
                });

                this.classList.add('is-active');
                annotation.activeKeypointType = "new-point";
                console.log('Ready to place new keypoint');
            }
        });
    }
}

async function handleSamPointClick() {
    if (!annotation.firstInitialised) {
        // First click - initialize
        annotation.firstInitialised = true;
        annotation.hasLowMask = true;
        const point = createPointElement(app.mouseX, app.mouseY, 'green');
        svgElement.appendChild(point);
    } else {
        if (isPointInPolygon([app.mouseX, app.mouseY], annotation.annotationCanditate.mask_polygon)) {
            const point = createPointElement(app.mouseX, app.mouseY, 'red');
            svgElement.appendChild(point);
            annotation.accumulatedPoints.push([app.mouseX, app.mouseY])
            annotation.accumulatedLabels.push(0)
            annotation.annotationCanditate = await debouncedMaskRequest();
            await drawCandidateAnnotations(annotation.annotationCanditate);
            console.log("Sam point clicked inside polygon")
        } else {
            const point = createPointElement(app.mouseX, app.mouseY, 'green');
            svgElement.appendChild(point);
            annotation.accumulatedPoints.push([app.mouseX, app.mouseY])
            annotation.accumulatedLabels.push(1)
            annotation.annotationCanditate = await debouncedMaskRequest();
            await drawCandidateAnnotations(annotation.annotationCanditate);
            console.log("Sam point clicked outside polygon")
        }
    }
    manageLabelButtonsState();
}

function createPointElement(x, y, color) {
    const estimated_r = imageWidth / 1200 * 5;
    return SVGCircleElement(x, y, estimated_r, color);
}

function validateAnnotation() {
    // Get the selected label ID and color
    const selectedLabel = document.querySelector('.is-label-button.is-active');
    const labelId = selectedLabel ? selectedLabel.dataset.labelId : '0';
    const labelColor = selectedLabel ? selectedLabel.dataset.labelColor : '466565';

    // Denormalize SAM3 response from [0,1] to pixel coordinates
    const maskPolygon = annotation.annotationCanditate.mask_polygon;
    const denormPolygon = [];
    for (let i = 0; i < maskPolygon.length; i += 2) {
        denormPolygon.push(
            maskPolygon[i] * imageWidth,      // x
            maskPolygon[i + 1] * imageHeight  // y
        );
    }

    const bbox = annotation.annotationCanditate.bbox;
    const denormBbox = [
        bbox[0] * imageWidth,  // x_min -> pixel
        bbox[1] * imageHeight, // y_min -> pixel
        bbox[2] * imageWidth,  // x_max -> pixel
        bbox[3] * imageHeight  // y_max -> pixel
    ];

    const annotationOptions = {
        bbox: denormBbox,
        class: labelId,
        segmentation: denormPolygon,
        id: null,
        fill: '#' + labelColor,
        stroke: '#' + labelColor,
        uuid: annotation.annotationCanditate.uuid
    };

    markin.deleteSelectedElement(annotation.annotationCanditate)
    markin.createAnnotation(annotationOptions)

    // This will prevent the annotation from being deleted
    markin.deselect()
    annotation.annotationCanditate = null
    const exported = markin.exportAllAnnotations({
        normalize: true,
        width: imageWidth,
        height: imageHeight
    });
    updateAPIAnnotations(normalizeAnnotationsForStorage(exported));
    resetAnnotation();
}

function toggleSAM() {
    const samToggle = document.getElementById('samToggle');
    const samToggleIcon = document.getElementById('samToggleIcon');

    // Reset current annotation regardless of mode switch
    resetAnnotation();

    if (app.samEnabled) {
        // Switching from SAM mode to manual mode
        app.samEnabled = false;
        samToggle.classList.toggle('is-active');
        if (samToggleIcon) samToggleIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-6"> <path stroke-linecap="round" stroke-linejoin="round" d="M3.98 8.223A10.477 10.477 0 0 0 1.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.451 10.451 0 0 1 12 4.5c4.756 0 8.773 3.162 10.065 7.498a10.522 10.522 0 0 1-4.293 5.774M6.228 6.228 3 3m3.228 3.228 3.65 3.65m7.894 7.894L21 21m-3.228-3.228-3.65-3.65m0 0a3 3 0 1 0-4.243-4.243m4.242 4.242L9.88 9.88" /> </svg>';
        // Enable VivaSVG for manual annotation mode
        markin.enable();
    } else {
        // Switching from manual mode to SAM mode
        app.samEnabled = true;
        samToggle.classList.toggle('is-active');
        if (samToggleIcon) samToggleIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-6"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" /></svg>';
        // Disable VivaSVG during SAM mode
        markin.disable();
    }
    console.log("SAM mode:", app.samEnabled ? "enabled" : "disabled");
}

function resetAnnotation() {
    annotation.firstInitialised = false;
    annotation.annotationCanditate = null;
    annotation.hasLowMask = false;
    annotation.accumulatedPoints = [[app.mouseX, app.mouseY]];
    annotation.accumulatedLabels = [1];
    markin.deleteSelectedElement(annotation.annotationCanditate)
    // Remove circle elements used for point indication
    let circleElements = document.querySelectorAll('.circle-element');
    if (circleElements) {
        circleElements.forEach(function (element) {
            element.parentNode.removeChild(element);
        });
    }
}


function normalizeAnnotationsForStorage(exported) {
    if (!exported || !exported.objects) return exported;
    return {
        objects: exported.objects.map(obj => {
            if (!obj.keypoints) return obj;
            return {
                ...obj,
                keypoints: obj.keypoints.map(kp => {
                    const label = (typeof projectLabels !== 'undefined') ? projectLabels.find(l => l.id == obj.class) : null;
                    const kpDef = label?.keypoints?.find(k => k.name === kp.name);
                    return {
                        id: kpDef?.id ?? 0,
                        name: kp.name,
                        point: kp.point
                    };
                })
            };
        })
    };
}

async function updateAPIAnnotations(exported_annotations) {
    const imageId = document.querySelector('.zoomist-image img').dataset.imageId;
    const response = await fetch('/project_update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image_id: imageId,
            value: exported_annotations,
            type: 'update_annotations'
        })
    });
}

function SVGCircleElement(x, y, r = 10, fill = 'red') {
    // Create the circle element
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');

    // Normalize the coordinates
    const normalizedX = annotation.croppedBbox[0] + (x / app.zoomistScale);
    const normalizedY = annotation.croppedBbox[1] + (y / app.zoomistScale);

    r = r / app.zoomistScale;

    // Set the attributes for the circle
    circle.setAttribute('cx', normalizedX);
    circle.setAttribute('cy', normalizedY);
    circle.setAttribute('r', r);
    circle.setAttribute('fill', fill);
    circle.setAttribute('class', 'circle-element');
    circle.setAttribute('vector-effect', 'non-scaling-stroke');
    return circle;
}

// Check if point is in polygon
// point is in pixel coords; polygon is normalized [0,1]
function isPointInPolygon(point, polygon) {
    let x = point[0] / imageWidth;
    let y = point[1] / imageHeight;

    let isInside = false;
    for (let i = 0, j = polygon.length - 2; i < polygon.length; j = i, i += 2) {
        let xi = polygon[i], yi = polygon[i + 1];
        let xj = polygon[j], yj = polygon[j + 1];
        let intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) isInside = !isInside;
    }
    return isInside;
}

// Capture mousedown to record starting position
svgElement.addEventListener('mousemove', async function (e) {
    const currentBbox = annotation.croppedBbox

    // Handle embedding request if zoomed and bbox changed
    if (app.zoomistScale > 1) {
        if (hasBboxChanged(currentBbox, app.previousBbox)) {
            app.previousBbox = await debouncedEmbeddingRequest();
        }
    }

    // Always update mask regardless of zoom
    if (!annotation.firstInitialised && app.samEnabled) {
        annotation.accumulatedPoints = [[app.mouseX, app.mouseY]]
        annotation.annotationCanditate = await debouncedMaskRequest();
        if (annotation.annotationCanditate) {
            await drawCandidateAnnotations(annotation.annotationCanditate);
        }
    }
});

// Capture mousemove to detect dragging
document.addEventListener('mousemove', function (e) {
    if (app.dragStartX !== null && app.dragStartY !== null) {
        const dx = e.clientX - app.dragStartX;
        const dy = e.clientY - app.dragStartY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > MIN_DRAG_DISTANCE) {
            app.isDragging = true;
        }
    }
    checkMouseOnSVG(e)
}, true);

// Block VivaSVG selection if we've been dragging
svgElement.addEventListener('click', function (e) {
    if (app.isDragging) {
        e.stopPropagation();
        e.preventDefault();
        console.log("Selection blocked - detected as drag not click");
    }

    // Reset drag state
    app.isDragging = false;
    app.dragStartX = null;
    app.dragStartY = null;
}, true);

// Function to check if bbox has changed
function hasBboxChanged(currentBbox, previousBbox) {
    return !previousBbox || !currentBbox.every((val, idx) => val === previousBbox[idx]);
}

// Draw candidate annotations using VivaSVG
async function drawCandidateAnnotations(candidate_annotation) {
    const candidate_polygon = candidate_annotation.mask_polygon;
    const candidate_bbox = candidate_annotation.bbox;

    // Denormalize SAM3 response from [0,1] to pixel coordinates
    // candidate_polygon is a flat array: [x1, y1, x2, y2, ...]
    const denormPolygon = [];
    for (let i = 0; i < candidate_polygon.length; i += 2) {
        denormPolygon.push(
            candidate_polygon[i] * imageWidth,      // x
            candidate_polygon[i + 1] * imageHeight  // y
        );
    }

    const denormBbox = [
        candidate_bbox[0] * imageWidth,  // x_min -> pixel
        candidate_bbox[1] * imageHeight, // y_min -> pixel
        candidate_bbox[2] * imageWidth,  // x_max -> pixel
        candidate_bbox[3] * imageHeight  // y_max -> pixel
    ];

    // Get the selected label ID and color
    const selectedLabel = document.querySelector('.is-label-button.is-active');
    const labelId = selectedLabel ? selectedLabel.dataset.labelId : '0';
    const labelColor = selectedLabel ? selectedLabel.dataset.labelColor : '466565';

    // Check if annotation already exists
    const existingAnnotation = document.querySelector('#candidate-annotation');

    // Prepare annotation options with color information
    const annotationOptions = {
        bbox: denormBbox,
        class: labelId,
        segmentation: denormPolygon,
        id: 'candidate-annotation',
        fill: '#' + labelColor,
        stroke: '#' + labelColor,
        uuid: 'candidate-annotation'
    };
    if (existingAnnotation) {
        markin.updateAnnotation(annotationOptions);
    } else {
        markin.createAnnotation(annotationOptions);
    }

    return true;
}


function updateMousePosition(e) {
    // Get mouse position relative to SVG in browser coordinates
    const rectSvgElement = svgElement.getBoundingClientRect();
    const rectZoomistContainer = zoomistContainer.getBoundingClientRect();

    const zoomistContainerRelativeX = e.clientX - rectZoomistContainer.left;
    const zoomistContainerrelativeY = e.clientY - rectZoomistContainer.top;

    const browserZoomistX = e.clientX - rectZoomistContainer.left;
    const browserZoomistY = e.clientY - rectZoomistContainer.top;

    const browserSvgX = e.clientX - rectSvgElement.left;
    const browserSvgY = e.clientY - rectSvgElement.top;

    // Calculate scaling factors
    const scaleSvgX = imageWidth / rectSvgElement.width;
    const scaleSvgY = imageHeight / rectSvgElement.height;

    // Calculate scaling factors
    const scaleZoomistX = imageWidth / rectZoomistContainer.width;
    const scaleZoomistY = imageHeight / rectZoomistContainer.height;

    // Convert browser coordinates to SVG coordinates
    const svgX = browserSvgX * scaleSvgX;
    const svgY = browserSvgY * scaleSvgY;

    // Convert browser coordinates to SVG coordinates
    const mouseX = browserZoomistX * scaleZoomistX;
    const mouseY = browserZoomistY * scaleZoomistY;

    // Update global state with mouse coordinates
    app.mouseX = Math.round(mouseX);
    app.mouseY = Math.round(mouseY);
    app.svgX = Math.round(svgX);
    app.svgY = Math.round(svgY);
}

// Function to update guideline positions
function updateGuidelines() {
    const verticalLine = document.getElementById('guide-line-vertical');
    const horizontalLine = document.getElementById('guide-line-horizontal');

    const dashArrayX = 3 / app.zoomistScale;
    const dashArrayY = 3 / app.zoomistScale;

    verticalLine.setAttribute("x1", app.svgX);
    verticalLine.setAttribute("y1", 0);
    verticalLine.setAttribute("x2", app.svgX);
    verticalLine.setAttribute("y2", imageHeight);
    verticalLine.setAttribute("stroke-width", 1 / app.zoomistScale);
    verticalLine.setAttribute("stroke-dasharray", `${dashArrayX},${dashArrayY}`);

    horizontalLine.setAttribute("x1", 0);
    horizontalLine.setAttribute("y1", app.svgY);
    horizontalLine.setAttribute("x2", imageWidth);
    horizontalLine.setAttribute("y2", app.svgY);
    horizontalLine.setAttribute("stroke-width", 1 / app.zoomistScale);
    horizontalLine.setAttribute("stroke-dasharray", `${dashArrayX},${dashArrayY}`);
}

function manageLabelButtonsState() {
    if (annotation.firstInitialised) {
        document.getElementById('reset-annotations').disabled = false;
        document.getElementById('validate-annotations').disabled = false;
    } else {
        document.getElementById('reset-annotations').disabled = true;
        document.getElementById('validate-annotations').disabled = true;
    }
}

// Function to handle label button clicks
function handleLabelButtonClick(event) {
    // Get all label buttons
    const labelButtons = document.querySelectorAll('.is-label-button');

    // Remove active class from all buttons
    labelButtons.forEach(button => {
        button.classList.remove('is-active');
    });

    // Add active class to clicked button
    const clickedButton = event.currentTarget;
    clickedButton.classList.add('is-active');

    // Update URL with selected label ID
    const url = new URL(window.location.href);
    url.searchParams.set('selected_label', clickedButton.dataset.labelId);
    window.history.pushState({}, '', url);

    // Update URLs in sidebar project image list
    const sidebarLinks = document.querySelectorAll('#sidebar-project-image-list a');
    sidebarLinks.forEach(link => {
        const linkUrl = new URL(link.href);
        linkUrl.searchParams.set('selected_label', clickedButton.dataset.labelId);
        link.href = linkUrl.toString();
    });

    // For keypoint-detection, show only the selected label's keypoint group
    if (projectType === 'keypoint-detection') {
        const labelId = clickedButton.dataset.labelId;
        document.querySelectorAll('.keypoint-group').forEach(group => {
            group.style.display = group.dataset.labelId === labelId ? '' : 'none';
        });
        annotation.activeKeypointType = null;
        document.querySelectorAll('.is-keypoint-button').forEach(b => b.classList.remove('is-active'));
    }
}

function updateCroppedBbox() {
    const image2ContainerRatioX = imageWidth / zoomistContainerWidth;
    const image2ContainerRatioY = imageHeight / zoomistContainerHeight;

    const relativeX = Math.abs(imageContainer.getBoundingClientRect().left - zoomistContainer.getBoundingClientRect().left);
    const relativeY = Math.abs(imageContainer.getBoundingClientRect().top - zoomistContainer.getBoundingClientRect().top);

    const xmin = Math.round(relativeX * image2ContainerRatioX / app.zoomistScale);
    const ymin = Math.round(relativeY * image2ContainerRatioY / app.zoomistScale);
    const bboxX = (zoomistContainerWidth * image2ContainerRatioX / app.zoomistScale);
    const bboxY = (zoomistContainerHeight * image2ContainerRatioY / app.zoomistScale);
    const xmax = Math.round(xmin + bboxX);
    const ymax = Math.round(ymin + bboxY);

    const bbox = [xmin, ymin, xmax, ymax]
    annotation.croppedBbox = bbox;
}

// Function to handle embedding request with debounce
function debouncedEmbeddingRequest() {
    return new Promise((resolve) => {
        // Clear any existing timeout
        if (window.embeddingTimeout) {
            clearTimeout(window.embeddingTimeout);
        }

        // Set a new timeout to wait for 0.1 seconds of no changes
        window.embeddingTimeout = setTimeout(async () => {
            const embeddingResult = await requestNewEmbedding(imageId, annotation.croppedBbox, imageShape);
            resolve(embeddingResult); // Resolve with the new bbox
        }, 500);
    });
}

// Function to handle mask request with debounce
function debouncedMaskRequest() {

    return new Promise((resolve) => {
        // Clear any existing timeout
        if (window.maskTimeout) {
            clearTimeout(window.maskTimeout);
        }

        // Set a new timeout to wait for 50ms of no changes
        window.maskTimeout = setTimeout(async () => {
            const maskResult = await requestMask();
            resolve(maskResult);
        }, 50);
    });
}

let _maskAbortController = null;

async function requestMask() {
    if (!_samModelReady) return null;
    // Cancel any in-flight request to prevent pileup during rapid mouse movement
    if (_maskAbortController) _maskAbortController.abort();
    _maskAbortController = new AbortController();

    try {
        const response = await fetch('/api/sam/predict_points', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            signal: _maskAbortController.signal,
            body: JSON.stringify({
                'points': annotation.accumulatedPoints,
                'labels': annotation.accumulatedLabels.map(l => l === 1),
                'file_id': imageId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        // EfficientTAM API returns { segmentation, bbox, score }, but UI expects mask_polygon
        const maskData = result.data.masks[0];
        if (!maskData) return null;
        return {
            mask_polygon: maskData.segmentation,
            bbox: maskData.bbox,
            uuid: null
        };
    } catch (error) {
        if (error.name === 'AbortError') return null;
        console.error('There was a problem sending the coordinates:', error);
        return null;
    }
}

async function requestNewEmbedding() {
    if (!_samModelReady) return null;
    try {
        const response = await fetch('/api/sam/create_embedding', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                'file_id': imageId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return annotation.croppedBbox;
    } catch (error) {
        console.error('There was a problem while creating the embedding:', error);
    }
}


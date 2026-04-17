/**
 * Project creation form — dynamic label and keypoint builder.
 * Used by project_new.html.
 */
var projectTypeSelect = document.getElementById('project-type');

document.addEventListener('DOMContentLoaded', function() {
    addLabel();
    projectTypeSelect.addEventListener('change', onTypeChange);
});

function onTypeChange() {
    var isKeypoint = projectTypeSelect.value === 'keypoint-detection';
    document.querySelectorAll('.keypoint-editor').forEach(function(el) {
        el.style.display = isKeypoint ? '' : 'none';
    });
}

function addLabel(name, color) {
    var container = document.getElementById('labels-container');
    var isKeypoint = projectTypeSelect.value === 'keypoint-detection';
    var wrapper = document.createElement('div');
    wrapper.className = 'label-row-wrapper mb-2';
    wrapper.innerHTML =
        '<div class="field has-addons mt-2 mb-0">' +
        '<div class="control is-expanded">' +
        '<input class="input label-name" type="text" placeholder="Label name" value="' + (name || '') + '">' +
        '</div>' +
        '<div class="control">' +
        '<input class="label-color" type="color" value="' + (color || '#3498db') + '" style="height:40px; width:50px; border:1px solid #dbdbdb; border-radius:4px; cursor:pointer;">' +
        '</div>' +
        '<div class="control">' +
        '<button type="button" class="button is-light" onclick="this.closest(\'.label-row-wrapper\').remove()">&times;</button>' +
        '</div>' +
        '</div>' +
        '<div class="keypoint-editor pl-4 pt-1" style="' + (isKeypoint ? '' : 'display:none') + '">' +
        '<div class="keypoints-list"></div>' +
        '<button type="button" class="button is-small is-light mt-1" onclick="addKeypointToLabel(this)">+ Add keypoint</button>' +
        '<div class="skeleton-editor mt-2">' +
        '<p class="is-size-7 has-text-grey mb-1">Skeleton connections</p>' +
        '<div class="skeleton-connections-list"></div>' +
        '<button type="button" class="button is-small is-light mt-1" onclick="addSkeletonConnection(this)">+ Add connection</button>' +
        '</div>' +
        '</div>';
    container.appendChild(wrapper);
}

function addKeypointToLabel(btn, kpName, kpColor) {
    var kpList = btn.closest('.keypoint-editor').querySelector('.keypoints-list');
    var kpRow = document.createElement('div');
    kpRow.className = 'field has-addons mt-1 mb-0 keypoint-row';
    kpRow.innerHTML =
        '<div class="control is-expanded">' +
        '<input class="input is-small kp-name" type="text" placeholder="Keypoint name (e.g. head)" value="' + (kpName || '') + '">' +
        '</div>' +
        '<div class="control">' +
        '<input class="kp-color" type="color" value="' + (kpColor || '#e74c3c') + '" style="height:32px; width:40px; border:1px solid #dbdbdb; border-radius:4px; cursor:pointer;">' +
        '</div>' +
        '<div class="control">' +
        '<button type="button" class="button is-small is-light" onclick="this.closest(\'.field\').remove()">&times;</button>' +
        '</div>';
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

document.querySelector('form').addEventListener('submit', function(e) {
    e.preventDefault();

    var labels = [];
    var isKeypoint = projectTypeSelect.value === 'keypoint-detection';
    var hasValidLabel = false;

    document.querySelectorAll('#labels-container .label-row-wrapper').forEach(function(wrapper, index) {
        var nameInput = wrapper.querySelector('.label-name');
        var colorInput = wrapper.querySelector('.label-color');
        var name = nameInput.value.trim();
        var color = colorInput.value.replace('#', '');

        if (name) {
            var labelObj = { id: index, name: name, color: color };

            if (isKeypoint) {
                var keypoints = [];
                wrapper.querySelectorAll('.keypoints-list .field').forEach(function(kpRow, kpIndex) {
                    var kpName = kpRow.querySelector('.kp-name').value.trim();
                    var kpColor = kpRow.querySelector('.kp-color').value.replace('#', '');
                    if (kpName) {
                        keypoints.push({ id: kpIndex, name: kpName, color: kpColor });
                    }
                });
                labelObj.keypoints = keypoints;
                var skeleton = [];
                wrapper.querySelectorAll('.skeleton-connections-list .skeleton-row').forEach(function (connRow) {
                    var from = parseInt(connRow.querySelector('.skeleton-from').value, 10);
                    var to = parseInt(connRow.querySelector('.skeleton-to').value, 10);
                    if (!isNaN(from) && !isNaN(to) && from !== to) skeleton.push([from, to]);
                });
                labelObj.skeleton = skeleton;
            }

            labels.push(labelObj);
            hasValidLabel = true;
        }
    });

    if (!hasValidLabel) {
        alert('Please add at least one valid label.');
        return;
    }

    document.getElementById('labels-json').value = JSON.stringify(labels);
    e.target.submit();
});

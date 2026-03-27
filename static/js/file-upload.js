/**
 * File Upload — drop zone and sequential file upload UI
 * Used by the project sidebar in project.html
 * Call initFileUpload(projectId) after the DOM is ready.
 */

function initFileUpload(projectId) {
    var dropZone = document.getElementById('drop-zone');
    var fileInput = document.getElementById('file-input');
    var uploadProgress = document.getElementById('upload-progress');
    var uploadBar = document.getElementById('upload-bar');
    var uploadStatus = document.getElementById('upload-status');
    var dropZoneText = document.getElementById('drop-zone-text');

    dropZone.addEventListener('click', function () {
        fileInput.click();
    });
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            uploadFiles(fileInput.files);
            fileInput.value = '';
        }
    });

    dropZone.addEventListener('dragenter', function (e) {
        e.preventDefault();
        dropZone.classList.add('is-dragover');
    });
    dropZone.addEventListener('dragover', function (e) {
        e.preventDefault();
    });
    dropZone.addEventListener('dragleave', function () {
        dropZone.classList.remove('is-dragover');
    });
    dropZone.addEventListener('drop', function (e) {
        e.preventDefault();
        dropZone.classList.remove('is-dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadFiles(e.dataTransfer.files);
        }
    });

    function uploadFiles(fileList) {
        var files = Array.from(fileList).filter(function (f) {
            return f.type.startsWith('image/');
        });
        if (files.length === 0) return;

        var total = files.length;
        var completed = 0;
        dropZoneText.classList.add('is-hidden');
        uploadProgress.classList.remove('is-hidden');
        uploadStatus.textContent = '0 / ' + total;
        uploadBar.value = 0;

        function uploadNext() {
            if (completed >= total) {
                uploadProgress.classList.add('is-hidden');
                dropZoneText.classList.remove('is-hidden');
                return;
            }
            var file = files[completed];
            var formData = new FormData();
            formData.append('file', file);

            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/project/' + projectId + '/upload');
            xhr.upload.onprogress = function (e) {
                if (e.lengthComputable) {
                    var filePct = e.loaded / e.total;
                    var overall = ((completed + filePct) / total) * 100;
                    uploadBar.value = overall;
                }
            };
            xhr.onload = function () {
                completed++;
                uploadBar.value = (completed / total) * 100;
                uploadStatus.textContent = completed + ' / ' + total;
                if (xhr.status === 201) {
                    var data = JSON.parse(xhr.responseText);
                    appendCard(data);
                }
                uploadNext();
            };
            xhr.onerror = function () {
                completed++;
                uploadNext();
            };
            xhr.send(formData);
        }

        uploadNext();
    }

    function appendCard(data) {
        var grid = document.getElementById('image-grid');
        var col = document.createElement('div');
        col.className = 'column is-2-fullhd is-3-desktop is-4-tablet is-6-mobile file-card';
        col.innerHTML =
            '<a href="/project/' + projectId + '/' + data.id + '" class="card">' +
            '<div class="card-image">' +
            '<figure class="image is-4by3">' +
            '<img src="/files/' + data.id + '?key=small" alt="' + data.filename + '" loading="lazy">' +
            '</figure>' +
            '</div>' +
            '</a>';
        grid.appendChild(col);
        document.getElementById('empty-state').classList.add('is-hidden');
    }
}

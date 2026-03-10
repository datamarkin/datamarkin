from io import BytesIO

from PIL import Image as PILImage
from queries import get_file_by_id
from flask import abort, request, send_file
from config import file_path
from thumbnails import PRESETS, get_or_create_thumb


def files_route(file_id):
    file_row = get_file_by_id(file_id)
    if not file_row:
        abort(404)
    filepath = file_path(file_row["filename"])
    if not filepath.exists():
        abort(404)

    key = request.args.get("key")
    if key is not None:
        if key not in PRESETS:
            abort(400)
        thumb = get_or_create_thumb(filepath, file_id, key)
        if isinstance(thumb, PILImage.Image):
            # On-demand: stream processed image from memory
            buf = BytesIO()
            thumb.save(buf, "JPEG", quality=85)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")
        else:
            # Saved thumbnail on disk
            return send_file(thumb, mimetype="image/jpeg")

    return send_file(filepath)
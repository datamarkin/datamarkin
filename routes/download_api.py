"""Download progress API — SSE stream and JSON status for model downloads."""

import json
import threading
import time

from flask import Blueprint, Response, jsonify

download_api = Blueprint("download_api", __name__, url_prefix="/api/downloads")

# Global download state: {name: {name, status, pct, error}}
_states = {}
_lock = threading.Lock()


def update_download_state(name, status="idle", pct=0, error=None):
    """Update the download state for a named model. Called by route modules."""
    with _lock:
        _states[name] = {"name": name, "status": status, "pct": pct, "error": error}


def clear_download_state(name):
    """Remove a completed download from the active state."""
    with _lock:
        _states.pop(name, None)


@download_api.route("/stream")
def stream():
    """SSE stream of active download states."""
    def generate():
        while True:
            with _lock:
                active = {k: v for k, v in _states.items() if v["status"] != "idle"}
            if active:
                data = json.dumps(list(active.values()))
                yield f"data: {data}\n\n"
            else:
                yield f"data: []\n\n"
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@download_api.route("/status")
def status():
    """JSON snapshot of all download states."""
    with _lock:
        return jsonify(list(_states.values()))

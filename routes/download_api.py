"""Download progress API — SSE stream and JSON status for model downloads."""

import json
import queue
import threading

from flask import Blueprint, Response, jsonify

download_api = Blueprint("download_api", __name__, url_prefix="/api/downloads")

# Global download state: {name: {name, status, pct, error}}
_states = {}
_lock = threading.Lock()

# Connected SSE clients: list of queue.Queue
_clients = []
_clients_lock = threading.Lock()


def _push_to_clients():
    """Push current active state snapshot to all connected SSE clients."""
    with _lock:
        active = [v for v in _states.values() if v["status"] != "idle"]
    data = json.dumps(active)
    with _clients_lock:
        for q in _clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass


def update_download_state(name, status="idle", pct=0, error=None):
    """Update the download state for a named model. Called by route modules."""
    with _lock:
        _states[name] = {"name": name, "status": status, "pct": pct, "error": error}
    _push_to_clients()


def clear_download_state(name):
    """Remove a completed download from the active state."""
    with _lock:
        _states.pop(name, None)
    _push_to_clients()


@download_api.route("/stream")
def stream():
    """SSE stream of active download states. Auto-closes after 30s idle."""
    q = queue.Queue(maxsize=50)
    with _clients_lock:
        _clients.append(q)

    def generate():
        try:
            with _lock:
                active = [v for v in _states.values() if v["status"] != "idle"]
            yield f"data: {json.dumps(active)}\n\n"

            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    return
        finally:
            with _clients_lock:
                _clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@download_api.route("/status")
def status():
    """JSON snapshot of all download states."""
    with _lock:
        return jsonify(list(_states.values()))

import multiprocessing
import sys

# Must be called before anything else in a frozen app so that
# multiprocessing worker processes (e.g. PyTorch DataLoader workers)
# exit cleanly instead of re-executing the main entry point.
multiprocessing.freeze_support()

# Worker mode: when the frozen app is invoked with "worker" as first arg,
# run the training worker instead of the GUI.
if len(sys.argv) > 1 and sys.argv[1] == "worker":
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-id", required=True)
    args = parser.parse_args()
    from scripts.training_worker import main as worker_main
    worker_main(args.training_id)
    sys.exit(0)

import threading
import webview
from app import create_app
from db import init_db
import config


def run_flask(app):
    app.run(host="127.0.0.1", port=config.FLASK_PORT, use_reloader=False, debug=False)


def _wait_for_server(port, timeout=15):
    """Block until Flask is accepting connections."""
    import socket
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)


if __name__ == "__main__":
    init_db()
    app = create_app()
    t = threading.Thread(target=run_flask, args=(app,), daemon=True)
    t.start()
    _wait_for_server(config.FLASK_PORT)

    from update_check import check_for_update
    threading.Thread(target=check_for_update, daemon=True).start()

    import task_queue

    window = webview.create_window(
        "Datamarkin",
        f"http://127.0.0.1:{config.FLASK_PORT}/projects",
        width=1280,
        height=800,
        min_size=(900, 600),
    )

    def _on_closing():
        if task_queue.has_active():
            try:
                result = window.evaluate_js(
                    "confirm('Tasks are still running. Closing now may lose progress. Close anyway?')"
                )
                if not result:
                    return False
            except Exception:
                pass
        return True

    window.events.closing += _on_closing
    webview.start()

    # Window closed — gracefully stop background tasks
    task_queue.shutdown(timeout=15)

import threading
import webview
from app import create_app
from db import init_db
import config


def run_flask(app):
    app.run(host="127.0.0.1", port=config.FLASK_PORT, use_reloader=False, debug=False)


if __name__ == "__main__":
    init_db()
    app = create_app()
    t = threading.Thread(target=run_flask, args=(app,), daemon=True)
    t.start()
    webview.create_window(
        "Datamarkin",
        f"http://127.0.0.1:{config.FLASK_PORT}/projects",
        width=1280,
        height=800,
        min_size=(900, 600),
    )
    webview.start()

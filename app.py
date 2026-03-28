import atexit
import os
import signal

from flask import Flask, render_template, request
from db import get_db, init_db
from routes.projects_page_route import (
    projects_page_route, project_new_page_route, project_upload_route,
    project_image_page_route, project_settings_route, project_pipeline_route,
    project_configuration_route, project_apply_split_route,
)
from routes.project_page_route import project_page_route
# from routes.settings_page_route import settings_page_route
from routes.files_route import files_route
from routes.api import api
from routes.efficienttam_api import efficienttam_api
from routes.training_route import training_api
from config import APP_NAME, APP_VERSION, ALLOWED_EXTENSIONS


def get_active_tab():
    """Determine active tab based on current request path."""
    try:
        path = request.path
    except RuntimeError:
        return "projects"  # Default when no request context (app startup)

    if path in ["/", "/projects"]:
        return "projects"
    elif path == "/model-zoo":
        return "model_zoo"
    elif path == "/workflows":
        return "workflows"
    elif path == "/settings":
        return "settings"
    elif path.startswith("/project/"):
        return "project_detail"
    return "projects"


def _kill_running_trainings() -> None:
    try:
        db = get_db()
        rows = db.execute(
            "SELECT pid FROM trainings WHERE status='running' AND pid IS NOT NULL"
        ).fetchall()
        db.close()
        for row in rows:
            try:
                os.kill(row["pid"], signal.SIGTERM)
            except Exception:
                pass
    except Exception:
        pass


atexit.register(_kill_running_trainings)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

    # Set Jinja2 globals
    app.jinja_env.globals.update({
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "allowed_media_extensions": ALLOWED_EXTENSIONS
        },
        "active_tab": get_active_tab(),
    })

    init_db()

    # Reconcile stale running trainings from a previous crash/restart
    db = get_db()
    rows = db.execute("SELECT id, pid FROM trainings WHERE status='running'").fetchall()
    for row in rows:
        alive = False
        if row["pid"]:
            try:
                os.kill(row["pid"], 0)
                alive = True
            except (ProcessLookupError, PermissionError):
                pass
        if not alive:
            db.execute(
                "UPDATE trainings SET status='failed', error='Process not found on startup', updated_at=datetime('now') WHERE id=?",
                (row["id"],),
            )
    db.commit()
    db.close()

    app.register_blueprint(api)
    app.register_blueprint(efficienttam_api)
    app.register_blueprint(training_api)

    @app.route("/")
    @app.route("/projects")
    def projects():
        return projects_page_route()

    @app.route("/project/new", methods=["GET", "POST"])
    def project_new():
        return project_new_page_route()

    @app.route("/project/<project_id>")
    def project(project_id):
        return project_page_route(project_id)

    # TODO this is not a good file upload route
    @app.route("/project/<project_id>/settings", methods=["POST"])
    def project_settings(project_id):
        return project_settings_route(project_id)

    @app.route("/project/<project_id>/pipeline", methods=["POST"])
    def project_pipeline(project_id):
        return project_pipeline_route(project_id)

    @app.route("/project/<project_id>/configuration", methods=["POST"])
    def project_configuration(project_id):
        return project_configuration_route(project_id)

    @app.route("/project/<project_id>/apply-split", methods=["POST"])
    def project_apply_split(project_id):
        return project_apply_split_route(project_id)

    @app.route("/project/<project_id>/upload", methods=["POST"])
    def project_upload(project_id):
        return project_upload_route(project_id)

    @app.route("/project/<project_id>/<file_id>")
    def project_image(project_id, file_id):
        return project_image_page_route(project_id, file_id)

    # @app.route("/settings")
    # def settings():
    #     return settings_page_route()

    @app.route("/model-zoo")
    def model_zoo():
        return render_template("model_zoo.html")

    @app.route("/agents")
    def agents():
        return render_template("agents.html")

    @app.route("/workflows")
    def workflows():
        return render_template("workflows.html")

    @app.route("/files/<file_id>")
    def serve_file(file_id):
        return files_route(file_id)

    return app

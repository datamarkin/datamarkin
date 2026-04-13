import atexit
import json
import os
import signal
import sys

from flask import Flask, abort, jsonify, redirect, render_template, request
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
from routes.predict_route import predict_api
from routes.falcon_perception_api import falcon_perception_api
from routes.download_api import download_api
from queries import (
    get_done_trainings_with_project, get_project_by_id, get_training,
    list_workflows, get_workflow_by_id, save_workflow, update_workflow, delete_workflow,
)
from config import APP_DIR, APP_NAME, APP_VERSION, ALLOWED_EXTENSIONS, DB_PATH


def get_active_tab():
    """Determine active tab based on current request path."""
    try:
        path = request.path
    except RuntimeError:
        return "projects"

    if path in ("/", "/projects"):
        return "projects"
    elif path in ("/workflows",) or path.startswith("/agentui"):
        return "workflows"
    elif path.startswith("/studio"):
        return "studio"
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
    if getattr(sys, "frozen", False):
        app = Flask(
            __name__,
            template_folder=str(APP_DIR / "templates"),
            static_folder=str(APP_DIR / "static"),
        )
    else:
        app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

    # Set Jinja2 globals
    app.jinja_env.globals.update({
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "allowed_media_extensions": ALLOWED_EXTENSIONS
        },
    })

    @app.context_processor
    def inject_active_tab():
        return {"active_tab": get_active_tab()}

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
    app.register_blueprint(predict_api)
    app.register_blueprint(falcon_perception_api)
    app.register_blueprint(download_api)

    # AgentUI visual workflow builder — mounted at /agentui
    try:
        import agentui
        from agentui.api.server import bp as agentui_bp
        from flask import jsonify as _jsonify, request as _request
        from tools.agentui_tools import DatamarkinLocalModel, METADATA as DM_METADATA
        agentui.register_tool(DatamarkinLocalModel, DM_METADATA)

        # In frozen app, the blueprint's template/static paths derived from __file__
        # are wrong because the module is loaded from the PYZ archive. Override with
        # the actual filesystem paths where PyInstaller placed the data files.
        if getattr(sys, "frozen", False):
            agentui_static = APP_DIR / "agentui" / "static"
            agentui_bp.template_folder = str(agentui_static)
            agentui_bp.static_folder = str(agentui_static / "assets")
        agentui.set_header(
            'agentui_header.html',
            context_fn=lambda: {'saved_workflows': list_workflows()}
        )

        # Workflow persistence routes — registered on the main app so they shadow
        # the agentui blueprint's cloud-proxy fallbacks when embedded here.
        @app.route("/agentui/api/workflows", methods=["GET", "POST"])
        def agentui_workflows():

            if _request.method == "POST":
                data = _request.get_json() or {}
                row = save_workflow(
                    name=data.get("name", "Untitled"),
                    description=data.get("description", ""),
                    workflow_json=json.dumps(data.get("workflow", {})),
                )
                return _jsonify(row), 201
            return _jsonify(list_workflows())

        @app.route("/agentui/api/workflows/<workflow_id>", methods=["GET", "PATCH", "DELETE"])
        def agentui_workflow(workflow_id):

            if _request.method == "GET":
                row = get_workflow_by_id(workflow_id)
                if row is None:
                    return _jsonify({"error": "Not found"}), 404
                result = dict(row)
                result["code"] = json.loads(result.pop("workflow_json", "{}"))
                return _jsonify(result)
            if _request.method == "PATCH":
                data = _request.get_json() or {}
                result = update_workflow(workflow_id, data)
                if result is None:
                    return _jsonify({"error": "Not found"}), 404
                return _jsonify(result)
            if _request.method == "DELETE":
                delete_workflow(workflow_id)
                return _jsonify({"ok": True})

        agentui_bp.url_prefix = '/agentui'
        app.register_blueprint(agentui_bp, url_prefix='/agentui')
    except ImportError as e:
        print(f"AgentUI not available: {e}")

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

    @app.route("/studio")
    def studio():
        trainings = []
        for t in get_done_trainings_with_project():
            t["config"] = json.loads(t.get("config") or "{}")
            t["metrics"] = json.loads(t.get("metrics") or "{}")
            trainings.append(t)
        workflows = list_workflows()
        return render_template("studio.html", trainings=trainings, workflows=workflows)

    @app.route("/studio/<training_id>")
    def studio_playground(training_id):
        training = get_training(training_id)
        if not training or training["status"] != "done":
            abort(404)
        training["config"] = json.loads(training.get("config") or "{}")
        training["metrics"] = json.loads(training.get("metrics") or "{}")
        project = get_project_by_id(training["project_id"])
        project_name = project["name"] if project else "Unknown"
        return render_template("studio_playground.html", mode="model", training=training, project_name=project_name, workflow=None)

    @app.route("/studio/workflow/<workflow_id>")
    def studio_workflow_playground(workflow_id):
        workflow = get_workflow_by_id(workflow_id)
        if not workflow:
            abort(404)
        workflow["workflow_json"] = json.loads(workflow.get("workflow_json") or "{}")
        return render_template(
            "studio_playground.html",
            mode="workflow",
            workflow=workflow,
            training=None,
            project_name=workflow["name"],
        )

    @app.route("/inference")
    def inference_redirect():
        return redirect("/studio")

    @app.route("/workflows")
    def workflows():
        return redirect("/agentui/")

    @app.route("/files/<file_id>")
    def serve_file(file_id):
        return files_route(file_id)

    # ── Update check API ────────────────────────────────────────────────
    from update_check import get_update_info, download_update

    @app.route("/api/update-check")
    def update_check():
        info = get_update_info()
        if info:
            return jsonify({"available": True, **info})
        return jsonify({"available": False})

    @app.route("/api/update-download", methods=["POST"])
    def update_download():
        try:
            path = download_update()
            if path:
                return jsonify({"ok": True, "path": path})
            return jsonify({"ok": False, "error": "No download available"}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return app

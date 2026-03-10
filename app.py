from flask import Flask, render_template, request
from db import init_db
from routes.projects_page_route import projects_page_route, project_new_page_route, project_upload_route, \
    project_image_page_route
from routes.project_page_route import project_page_route
from routes.settings_page_route import settings_page_route
from routes.files_route import files_route
from routes.api import api
from routes.sam3_api import sam3_api
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
    app.register_blueprint(api)
    app.register_blueprint(sam3_api)

    @app.route("/")
    @app.route("/projects")
    def projects():
        return projects_page_route("Datamarkin")

    @app.route("/project/new", methods=["GET", "POST"])
    def project_new():
        return project_new_page_route()

    @app.route("/project/<project_id>")
    def project(project_id):
        return project_page_route(project_id)

    # TODO this is not a good file upload route
    @app.route("/project/<project_id>/upload", methods=["POST"])
    def project_upload(project_id):
        return project_upload_route(project_id)

    @app.route("/project/<project_id>/<file_id>")
    def project_image(project_id, file_id):
        return project_image_page_route(project_id, file_id)

    @app.route("/settings")
    def settings():
        return settings_page_route()

    @app.route("/model-zoo")
    def model_zoo():
        return render_template("model_zoo.html")

    @app.route("/workflows")
    def workflows():
        return render_template("workflows.html")

    @app.route("/files/<file_id>")
    def serve_file(file_id):
        return files_route(file_id)

    return app

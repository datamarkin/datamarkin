from flask import Flask, render_template, send_file, abort, request
from config import file_path
from thumbnails import PRESETS, get_or_create_thumb

from db import init_db
from queries import get_file_by_id
from routes.projects_page_route import projects_page_route, project_new_page_route, project_page_route, project_image_page_route


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

    init_db()

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

    @app.route("/project/<project_id>/annotate/<file_id>")
    def project_image(project_id, file_id):
        return project_image_page_route(project_id, file_id)

    @app.route("/model-zoo")
    def model_zoo():
        return render_template("model_zoo.html", active_tab="model_zoo")

    @app.route("/workflows")
    def workflows():
        return render_template("workflows.html", active_tab="workflows")

    @app.route("/files/<file_id>")
    def serve_file(file_id):
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
            return send_file(thumb, mimetype="image/jpeg")

        return send_file(filepath)

    return app

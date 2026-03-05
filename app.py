from flask import Flask, render_template, send_file, abort
from config import DATA_DIR

from db import init_db
from routes.projects_page_route import projects_page_route, project_detail_route


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

    init_db()

    @app.route("/")
    @app.route("/projects")
    def projects():
        return projects_page_route("Datamarkin")

    @app.route("/project/<project_id>")
    def project_detail(project_id):
        return project_detail_route(project_id)

    @app.route("/model-zoo")
    def model_zoo():
        return render_template("model_zoo.html", active_tab="model_zoo")

    @app.route("/workflows")
    def workflows():
        return render_template("workflows.html", active_tab="workflows")

    @app.route("/projects/<project_id>/images/<filename>")
    def serve_project_image(project_id, filename):
        filepath = DATA_DIR / "projects" / project_id / "images" / filename
        if not filepath.exists():
            abort(404)
        return send_file(filepath)

    return app

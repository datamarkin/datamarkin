from flask import Flask, render_template
from db import init_db
from routes.projects_page_route import projects_page_route,project_new_page_route, project_upload_route, project_image_page_route
from routes.project_page_route import project_page_route
from routes.settings_page_route import settings_page_route
from routes.files_route import files_route
from routes.api import api
from routes.sam3_api import sam3_api


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

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

    #TODO this is not a good file upload route
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
        return render_template("model_zoo.html", active_tab="model_zoo")

    @app.route("/workflows")
    def workflows():
        return render_template("workflows.html", active_tab="workflows")

    @app.route("/files/<file_id>")
    def serve_file(file_id):
        return files_route(file_id)

    return app

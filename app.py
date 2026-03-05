from flask import Flask
from routes.projects_page_route import projects_page_route


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

    @app.route("/")
    @app.route("/projects")
    def projects():
        return projects_page_route("Datamarkin")
    return app
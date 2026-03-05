from flask import render_template

def projects_page_route(app_name):
    return render_template("projects.html", app_name="Datamarkin", active_tab="projects")

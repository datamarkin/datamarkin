"""Task queue API — unified view of all background operations."""

from flask import Blueprint, jsonify

import task_queue

task_api = Blueprint("task_api", __name__, url_prefix="/api/tasks")


@task_api.route("/", methods=["GET"])
def list_tasks():
    return jsonify(task_queue.get_tasks())


@task_api.route("/<task_id>/cancel", methods=["POST"])
def cancel_task(task_id):
    if task_queue.cancel(task_id):
        return jsonify({"ok": True})
    return jsonify({"error": "Task not found or already finished"}), 404


@task_api.route("/active", methods=["GET"])
def active_check():
    return jsonify({"active": task_queue.has_active()})

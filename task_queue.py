"""Centralized task queue for background operations.

Executor functions are the plugin interface — just ``fn(ctx: TaskContext) -> None``.
The queue owns thread lifecycle, progress tracking, cancellation, and shutdown.

Usage::

    import task_queue

    def my_executor(ctx):
        for i, item in enumerate(items):
            if ctx.is_cancelled():
                return
            do_work(item)
            ctx.progress((i + 1) / len(items), f"{i + 1}/{len(items)}")

    task_queue.submit("my_type", my_executor, label="Working…", meta={…})
"""

import threading
import time
import uuid


# ---------------------------------------------------------------------------
# TaskContext — the stable interface passed to executor functions
# ---------------------------------------------------------------------------

class TaskContext:
    """Passed to executor functions. Provides progress reporting and
    cooperative cancellation."""

    __slots__ = ("task_id", "meta", "_cancel_event", "_update_fn")

    def __init__(self, task_id: str, meta: dict, cancel_event: threading.Event):
        self.task_id = task_id
        self.meta = meta
        self._cancel_event = cancel_event
        self._update_fn = None  # set by _run_executor

    def progress(self, pct: float, detail: str = ""):
        """Report progress.  *pct* is 0.0 – 1.0."""
        if self._update_fn:
            self._update_fn(progress=pct, detail=detail)

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_event.is_set()


# ---------------------------------------------------------------------------
# Internal task record
# ---------------------------------------------------------------------------

class _Task:
    __slots__ = (
        "id", "type", "label", "status", "progress", "detail", "error",
        "executor_fn", "cancel_event", "thread", "meta",
        "created_at", "started_at", "finished_at", "max_concurrency",
    )

    def __init__(self, task_id, task_type, label, executor_fn, meta, max_concurrency):
        self.id = task_id
        self.type = task_type
        self.label = label
        self.status = "queued"
        self.progress = 0.0
        self.detail = ""
        self.error = None
        self.executor_fn = executor_fn
        self.cancel_event = threading.Event()
        self.thread = None
        self.meta = meta or {}
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.max_concurrency = max_concurrency

    def to_dict(self, include_meta=False):
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "status": self.status,
            "progress": self.progress,
            "detail": self.detail,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if include_meta:
            d["meta"] = self.meta
        return d


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_tasks: dict[str, _Task] = {}
_lock = threading.Lock()

_PRUNE_AGE = 120  # seconds after finish before pruning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit(task_type: str, executor_fn, *, label: str,
           meta: dict | None = None, max_concurrency: int = 1) -> str:
    """Submit a task for background execution.  Returns the task id."""
    task_id = uuid.uuid4().hex[:12]
    task = _Task(task_id, task_type, label, executor_fn, meta, max_concurrency)
    with _lock:
        _tasks[task_id] = task
    _try_dispatch(task_type)
    return task_id


def cancel(task_id: str) -> bool:
    """Request cancellation of a task.  Returns True if the task was found."""
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return False
        if task.status == "queued":
            task.status = "cancelled"
            task.finished_at = time.time()
            return True
        if task.status == "running":
            task.cancel_event.set()
            return True
    return False


def get_tasks() -> list[dict]:
    """Return all tasks (most-recent first), pruning old finished ones."""
    now = time.time()
    with _lock:
        # prune
        to_remove = [
            tid for tid, t in _tasks.items()
            if t.finished_at and (now - t.finished_at) > _PRUNE_AGE
        ]
        for tid in to_remove:
            del _tasks[tid]
        return [t.to_dict() for t in
                sorted(_tasks.values(), key=lambda t: t.created_at, reverse=True)]


def has_active() -> bool:
    """Return True if any task is queued or running."""
    with _lock:
        return any(t.status in ("queued", "running") for t in _tasks.values())


def find_task(task_type: str | None = None, **meta_match) -> dict | None:
    """Find the first active task matching *task_type* and/or meta keys."""
    with _lock:
        for t in _tasks.values():
            if t.status not in ("queued", "running"):
                continue
            if task_type and t.type != task_type:
                continue
            if all(t.meta.get(k) == v for k, v in meta_match.items()):
                return t.to_dict(include_meta=True)
    return None


def find_recent_task(task_type: str) -> dict | None:
    """Find the most recent task of *task_type* (any status, not yet pruned)."""
    with _lock:
        best = None
        for t in _tasks.values():
            if t.type != task_type:
                continue
            if best is None or t.created_at > best.created_at:
                best = t
        return best.to_dict(include_meta=True) if best else None


def shutdown(timeout: float = 15):
    """Cancel all active tasks and wait for running ones to finish."""
    with _lock:
        active = [t for t in _tasks.values() if t.status in ("queued", "running")]
        for t in active:
            if t.status == "queued":
                t.status = "cancelled"
                t.finished_at = time.time()
            else:
                t.cancel_event.set()

    deadline = time.time() + timeout
    for t in active:
        if t.thread and t.thread.is_alive():
            remaining = max(0, deadline - time.time())
            t.thread.join(timeout=remaining)


# ---------------------------------------------------------------------------
# Internal dispatch / execution
# ---------------------------------------------------------------------------

def _try_dispatch(task_type: str):
    """Start the next queued task of *task_type* if concurrency allows."""
    with _lock:
        running = sum(1 for t in _tasks.values()
                      if t.type == task_type and t.status == "running")

        queued = sorted(
            (t for t in _tasks.values()
             if t.type == task_type and t.status == "queued"),
            key=lambda t: t.created_at,
        )
        if not queued:
            return

        max_c = queued[0].max_concurrency
        if running >= max_c:
            return

        task = queued[0]
        task.status = "running"
        task.started_at = time.time()

        thread = threading.Thread(target=_run_executor, args=(task,), daemon=True)
        task.thread = thread
        thread.start()


def _run_executor(task: _Task):
    """Thread target — runs the executor and updates task status."""
    ctx = TaskContext(task.id, task.meta, task.cancel_event)

    def _update(**kwargs):
        with _lock:
            for k, v in kwargs.items():
                if hasattr(task, k):
                    setattr(task, k, v)

    ctx._update_fn = _update

    try:
        task.executor_fn(ctx)
        with _lock:
            if task.cancel_event.is_set():
                task.status = "cancelled"
            else:
                task.status = "done"
                task.progress = 1.0
    except Exception as e:
        with _lock:
            task.status = "failed"
            task.error = str(e)
    finally:
        with _lock:
            task.finished_at = time.time()
        _try_dispatch(task.type)

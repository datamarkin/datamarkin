"""
Standalone training subprocess for Datamarkin.

Usage:
    python scripts/training_worker.py --training-id <id>
"""
import argparse
import json
import os
import shutil
import signal
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from config import DB_PATH, MODELS_DIR


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_status(training_id: str, status: str, error: str | None = None) -> None:
    conn = _db()
    conn.execute(
        "UPDATE trainings SET status=?, error=?, updated_at=? WHERE id=?",
        (status, error, _now(), training_id),
    )
    conn.commit()
    conn.close()


def _write_progress(training_id: str, progress: dict) -> None:
    conn = _db()
    conn.execute(
        "UPDATE trainings SET progress=?, updated_at=? WHERE id=?",
        (json.dumps(progress), _now(), training_id),
    )
    conn.commit()
    conn.close()


def _save_best_checkpoint(training_id: str, output_dir: str) -> str:
    """Copy best checkpoint from rfdetr output_dir to MODELS_DIR/<training_id>.pth."""
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / f"{training_id}.pth"
    best_src = Path(output_dir) / "checkpoint_best_total.pth"
    last_src = Path(output_dir) / "checkpoint.pth"
    src = best_src if best_src.exists() else last_src
    if src.exists():
        shutil.copy2(src, dest)
        return str(dest)
    return ""


def main(training_id: str) -> None:
    import torch

    # Read config from DB
    conn = _db()
    row = conn.execute("SELECT config FROM trainings WHERE id=?", (training_id,)).fetchone()
    conn.close()
    if not row:
        print(f"[worker] Training {training_id} not found in DB", flush=True)
        sys.exit(1)

    cfg = json.loads(row["config"])

    model_size          = cfg.get("model_size", "base")
    epochs              = cfg.get("epochs", 20)
    batch_size          = cfg.get("batch_size", 4)
    resolution          = cfg.get("resolution", 560)
    lr                  = cfg.get("lr", 1e-4)
    early_stopping      = cfg.get("early_stopping", True)
    early_stopping_pat  = cfg.get("early_stopping_patience", 3)
    dataset_dir         = cfg["dataset_dir"]

    # Select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Import rfdetr model class
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall
    MODEL_CLASSES = {
        "small": RFDETRSmall,
        "base":  RFDETRBase,
        "large": RFDETRLarge,
    }
    ModelClass = MODEL_CLASSES.get(model_size, RFDETRBase)

    output_dir = str(MODELS_DIR / training_id)

    stopped = {"flag": False}

    def _on_sigterm(signum, frame):
        stopped["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    model = ModelClass(resolution=resolution)
    history = []

    def on_epoch_end(data):
        if stopped["flag"]:
            return

        epoch_data = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()
        } if isinstance(data, dict) else {"raw": data}

        history.append(epoch_data)

        progress = {
            "epoch":        len(history),
            "total_epochs": epochs,
            **{k: v for k, v in epoch_data.items() if k != "epoch"},
        }
        _write_progress(training_id, progress)
        print(f"[worker] epoch {len(history)}/{epochs}: {epoch_data}", flush=True)

        if device == "mps":
            torch.mps.empty_cache()

    model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

    _set_status(training_id, "running")

    try:
        model.train(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_interval=5,
            gradient_checkpointing=True,
            multi_scale=False,
            use_ema=True,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_pat,
            device=device,
        )

        if stopped["flag"]:
            _save_best_checkpoint(training_id, output_dir)
            _set_status(training_id, "stopped")
            return

        model_path = _save_best_checkpoint(training_id, output_dir)

        best_map = max((e.get("map", 0) for e in history), default=None)
        final_loss = history[-1].get("loss") if history else None
        metrics = {
            "best_mAP":   best_map,
            "final_loss": final_loss,
            "history":    history,
        }

        conn = _db()
        conn.execute(
            "UPDATE trainings SET status='done', model_path=?, metrics=?, updated_at=? WHERE id=?",
            (model_path, json.dumps(metrics), _now(), training_id),
        )
        conn.commit()
        conn.close()

    except Exception:
        error_msg = traceback.format_exc()
        print(f"[worker] Training failed:\n{error_msg}", flush=True)
        _set_status(training_id, "failed", error=error_msg)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-id", required=True)
    args = parser.parse_args()
    main(args.training_id)

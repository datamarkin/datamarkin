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
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import config (not needed in frozen app)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pytorch_lightning
from pytorch_lightning import Callback

# PyInstaller fix: in the frozen app, pytorch_lightning.__version__ is never set
# because the os.path.isfile() checks in __init__.py fail inside the archive.
# Without this, rfdetr's `from pytorch_lightning import __version__` falls back
# to importing the __version__ submodule (a module object) which torch.save()
# cannot pickle during checkpoint saves.
if not hasattr(pytorch_lightning, "__version__") or not isinstance(pytorch_lightning.__version__, str):
    try:
        _ver_mod = __import__("pytorch_lightning.__version__", fromlist=["version"])
        pytorch_lightning.__version__ = _ver_mod.version
    except Exception:
        pytorch_lightning.__version__ = "unknown"

import pixelflow as pf

from config import DATA_DIR, DB_PATH, MODELS_DIR, TRAINING_JOBS_DIR
from pipeline import to_rfdetr_aug_config

RFDETR_ASSETS = {
    ("detection", "small"):      "rfdetr/rf-detr-small.pth",
    ("detection", "base"):       "rfdetr/rf-detr-base.pth",
    ("detection", "large"):      "rfdetr/rf-detr-large.pth",
    ("segmentation", "small"):   "rfdetr/rf-detr-seg-small.pt",
    ("segmentation", "base"):    "rfdetr/rf-detr-seg-medium.pt",
    ("segmentation", "large"):   "rfdetr/rf-detr-seg-large.pt",
}


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


def _write_metrics(training_id: str, metrics: dict) -> None:
    conn = _db()
    conn.execute(
        "UPDATE trainings SET metrics=?, updated_at=? WHERE id=?",
        (json.dumps(metrics), _now(), training_id),
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


class WorkerMetricsCallback(Callback):
    def __init__(self, live_path, epochs, stopped, project_type, history, training_id, device):
        self._last_live = 0.0
        self.live_path = live_path
        self.epochs = epochs
        self.stopped = stopped
        self.project_type = project_type
        self.history = history
        self.training_id = training_id
        self.device = device

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        if now - self._last_live < 5:
            return
        self._last_live = now

        loss_val = trainer.callback_metrics.get("train/loss")
        if loss_val is None and isinstance(outputs, torch.Tensor):
            loss_val = float(outputs) * max(1, trainer.accumulate_grad_batches)
        if loss_val is None:
            return

        try:
            self.live_path.write_text(json.dumps({
                "step":         trainer.global_step,
                "epoch":        trainer.current_epoch + 1,
                "total_epochs": self.epochs,
                "batch_loss":   float(loss_val),
                "timestamp":    now,
            }))
        except Exception:
            pass

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stopped["flag"]:
            trainer.should_stop = True
            return

        m = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in trainer.callback_metrics.items()
        }
        if self.project_type == "segmentation":
            map_val     = m.get("val/segm_mAP_50")
            map_50_95   = m.get("val/segm_mAP_50_95")
            ema_map_val = m.get("val/ema_segm_mAP_50")
        else:
            map_val     = m.get("val/mAP_50")
            map_50_95   = m.get("val/mAP_50_95")
            ema_map_val = m.get("val/ema_mAP_50")

        epoch_data = {
            "loss":            m.get("train/loss_epoch", m.get("train/loss")),
            "map":             map_val,
            "map_50_95":       map_50_95,
            "map_75":          m.get("val/mAP_75"),
            "recall":          m.get("val/mAR"),
            "f1":              m.get("val/F1"),
            "precision":       m.get("val/precision"),
            "ema_map_50":      ema_map_val,
            "segm_map_50":     m.get("val/segm_mAP_50"),
            "segm_map_50_95":  m.get("val/segm_mAP_50_95"),
        }
        epoch_data = {k: v for k, v in epoch_data.items() if v is not None}

        self.history.append(epoch_data)
        progress = {"epoch": len(self.history), "total_epochs": self.epochs, **epoch_data}
        _write_progress(self.training_id, progress)
        _write_metrics(self.training_id, {"history": self.history})
        print(f"[worker] epoch {len(self.history)}/{self.epochs}: {epoch_data}", flush=True)

        if self.device == "mps":
            torch.mps.empty_cache()


def main(training_id: str) -> None:
    # Redirect all output to a log file immediately so errors are visible
    job_dir = TRAINING_JOBS_DIR / training_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "worker.log"
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

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
        project_type        = cfg.get("project_type", "detection")

        # Convert UI augmentation pipeline to RF-DETR format.
        # Empty list disables RF-DETR's built-in default augmentations.
        aug_config          = to_rfdetr_aug_config(cfg.get("augmentation"))

        # Select device and matching PTL accelerator
        if torch.backends.mps.is_available():
            device, accelerator = "mps", "mps"
        elif torch.cuda.is_available():
            device, accelerator = "cuda", "gpu"
        else:
            device, accelerator = "cpu", "cpu"

        # Import rfdetr model class
        from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall
        from rfdetr import RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge

        if project_type == "segmentation":
            # "base" in our config → RFDETRSegMedium (rfdetr has no SegBase)
            MODEL_CLASSES = {"small": RFDETRSegSmall, "base": RFDETRSegMedium, "large": RFDETRSegLarge}
            default_cls = RFDETRSegMedium
        else:
            MODEL_CLASSES = {"small": RFDETRSmall, "base": RFDETRBase, "large": RFDETRLarge}
            default_cls = RFDETRBase
        ModelClass = MODEL_CLASSES.get(model_size, default_cls)

        output_dir = str(MODELS_DIR / training_id)
        live_path = job_dir / "live.json"

        stopped = {"flag": False}

        def _on_sigterm(signum, frame):
            stopped["flag"] = True

        signal.signal(signal.SIGTERM, _on_sigterm)

        # Download pretrained weights from dtmfiles.com (skips if cached)
        asset_path = RFDETR_ASSETS.get((project_type, model_size))
        if asset_path:
            ckpt = pf.assets.download(asset_path, directory=DATA_DIR)
            model = ModelClass(resolution=resolution, pretrain_weights=str(ckpt))
        else:
            model = ModelClass(resolution=resolution)

        # Align num_classes to match the dataset (pretrained model defaults to 90 COCO classes)
        model._align_num_classes_from_dataset(dataset_dir)

        history = []

    except Exception:
        error_msg = traceback.format_exc()
        print(f"[worker] Startup failed:\n{error_msg}", flush=True)
        _set_status(training_id, "failed", error=error_msg)
        sys.exit(1)

    _set_status(training_id, "running")

    try:
        config = model.get_train_config(
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
            aug_config=aug_config or {},
        )
        module     = RFDETRModelModule(model.model_config, config)
        datamodule = RFDETRDataModule(model.model_config, config)
        trainer    = build_trainer(config, model.model_config, accelerator=accelerator,
                                   num_sanity_val_steps=0)
        trainer.callbacks.append(WorkerMetricsCallback(
            live_path=live_path, epochs=epochs, stopped=stopped,
            project_type=project_type, history=history,
            training_id=training_id, device=device,
        ))

        trainer.fit(module, datamodule)

        # Sync trained weights back so predict() / export() work
        model.model.model = module.model

        if stopped["flag"]:
            _save_best_checkpoint(training_id, output_dir)
            _set_status(training_id, "stopped")
            return

        model_path = _save_best_checkpoint(training_id, output_dir)

        best_map   = max((e.get("map", 0) for e in history), default=None)
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

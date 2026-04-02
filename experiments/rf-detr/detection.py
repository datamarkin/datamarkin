import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # fallback unsupported MPS ops to CPU

import json
import time
import torch
from pytorch_lightning import Callback
from rfdetr import RFDETRNano
from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

# ── Config ───────────────────────────────────────────────────────────────────
# Dataset must be COCO format:
#   dataset/train/_annotations.coco.json + images
#   dataset/valid/_annotations.coco.json + images
#   dataset/test/_annotations.coco.json  + images
DATASET_DIR = "./dataset"
OUTPUT_DIR  = "output/detection"
EPOCHS      = 20
BATCH_SIZE  = 4
GRAD_ACCUM  = 4       # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
LR          = 1e-4
RESOLUTION  = 480      # must be divisible by 56 (560/616/672/728/784/840/896)


class MetricsLogger(Callback):
    """Logs per-epoch metrics to stdout and <output_dir>/metrics.jsonl."""

    def __init__(self, output_dir: str):
        self._jsonl_path = os.path.join(output_dir, "metrics.jsonl")
        self.epoch = 0
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.epoch += 1
        m = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in trainer.callback_metrics.items()
        }
        m["epoch"] = self.epoch
        m["timestamp"] = time.time()

        loss      = m.get("train/loss_epoch", m.get("train/loss", float("nan")))
        map_50    = m.get("val/mAP_50",      float("nan"))
        map_50_95 = m.get("val/mAP_50_95",   float("nan"))
        recall    = m.get("val/mAR",         float("nan"))
        f1        = m.get("val/F1",          float("nan"))
        print(
            f"[epoch {self.epoch:3d}]  "
            f"loss={loss:.4f}  mAP@50={map_50:.4f}  mAP@50:95={map_50_95:.4f}  "
            f"mAR={recall:.4f}  F1={f1:.4f}"
        )

        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(m) + "\n")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    # ── Model ─────────────────────────────────────────────────────────────────
    model = RFDETRNano(resolution=RESOLUTION)

    # ── Train (low-level PTL API to inject MetricsLogger callback) ────────────
    config = model.get_train_config(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        lr=LR,
        checkpoint_interval=5,
        gradient_checkpointing=True,  # recompute activations to save memory
        multi_scale=False,            # disable: default bumps 560px → 840px (2.25x memory)
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=3,
    )
    module     = RFDETRModelModule(model.model_config, config)
    datamodule = RFDETRDataModule(model.model_config, config)
    trainer    = build_trainer(config, model.model_config, accelerator="mps")
    trainer.callbacks.append(MetricsLogger(OUTPUT_DIR))
    trainer.fit(module, datamodule)

    # Sync trained weights back so predict() / export() work without reloading
    model.model.model = module.model

    print(f"Training complete. Best checkpoint: {OUTPUT_DIR}/checkpoint_best_total.pth")
    print(f"Metrics saved to: {OUTPUT_DIR}/metrics.jsonl")

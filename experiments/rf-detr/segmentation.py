import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # fallback unsupported MPS ops to CPU

import torch
from rfdetr import RFDETRSegNano

# ── Config ───────────────────────────────────────────────────────────────────
# Dataset must be COCO format with segmentation polygons:
#   annotations[i]["segmentation"] = [[x1, y1, x2, y2, ...]]
#   dataset/train/_annotations.coco.json + images
#   dataset/valid/_annotations.coco.json + images
#   dataset/test/_annotations.coco.json  + images
DATASET_DIR = "./COCO-Dataset-50"
OUTPUT_DIR  = "output/segmentation"
EPOCHS      = 20
BATCH_SIZE  = 1
GRAD_ACCUM  = 16       # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
LR          = 1e-4
RESOLUTION  = 600      # must be divisible by 56 (560/616/672/728/784/840/896)

if __name__ == "__main__":
    # ── Model ─────────────────────────────────────────────────────────────────
    model = RFDETRSegNano(resolution=RESOLUTION)

    # Collect per-epoch metrics as plain Python scalars (avoids holding GPU tensors)
    history = []

    def on_epoch_end(data):
        history.append({
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        } if isinstance(data, dict) else data)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()  # flush MPS memory pool each epoch

    model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train(
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
        # early_stopping disabled: rfdetr bug reloads 91-class COCO head after stopping,
        # mismatching the fine-tuned head. Checkpoints on disk are written correctly.
        device="mps",  # Apple Silicon; use "cuda" for NVIDIA, "cpu" as fallback
    )

    print(f"Segmentation training complete. Best checkpoint: {OUTPUT_DIR}/checkpoint_best_ema.pth")
    print("Per-epoch history:", history)

#!/usr/bin/env python3
"""Minimal Falcon-Perception demo on Apple Silicon (MLX backend).

Measures model loading, inference speed, and output quality for
text-prompted object detection and instance segmentation.

Install:
    cd /path/to/Falcon-Perception
    pip install -e ".[mlx]"

Usage:
    python demo.py
    python demo.py --image photo.jpg --query "cat"
    python demo.py --image photo.jpg --query "cat,person" --task both
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
)
from falcon_perception.data import load_image

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_IMAGE = str(Path(__file__).parent / "test.jpeg")
DEFAULT_QUERIES = ["person", "car", "bicycle", "window"]

_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]


# ── Helpers (inlined from upstream demo — not exported by package) ────

def pair_bbox_entries(raw: list[dict]) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...]."""
    bboxes, current = [], {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def decode_rle_mask(rle: dict) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle).astype(np.uint8)
    except Exception:
        return None


def visualize(
    image: Image.Image,
    bboxes: list[dict],
    masks_rle: list[dict],
    out_path: str,
    opacity: float = 0.35,
    border_px: int = 3,
):
    """Draw mask overlays and bounding boxes, save to out_path."""
    img = image.convert("RGB")
    W, H = img.size
    overlay = np.array(img, dtype=np.float32)

    masks = []
    for rle in masks_rle:
        m = decode_rle_mask(rle)
        if m is not None:
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
            masks.append(m)

    n_det = min(len(bboxes), len(masks)) if masks else len(bboxes)

    for i in range(min(n_det, len(masks))):
        m = masks[i]
        color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
        region = m > 0
        overlay[region] = overlay[region] * (1 - opacity) + color * opacity
        from scipy.ndimage import binary_dilation
        border = binary_dilation(region, iterations=border_px) & ~region
        overlay[border] = color

    result = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)

    for i, bbox in enumerate(bboxes[:n_det]):
        cx, cy = bbox["x"] * W, bbox["y"] * H
        bw, bh = bbox["w"] * W, bbox["h"] * H
        x0, y0 = cx - bw / 2, cy - bh / 2
        x1, y1 = cx + bw / 2, cy + bh / 2
        draw.rectangle([x0, y0, x1, y1], outline=_PALETTE[i % len(_PALETTE)], width=2)

    result.save(out_path)
    print(f"  Saved: {out_path}")


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.2f}s"


# ── Core inference ────────────────────────────────────────────────────

def run_inference(engine, tokenizer, model_args, pil_image, query, task, args):
    """Run a single inference pass. Returns dict with timing and results."""
    from falcon_perception.mlx.batch_inference import process_batch_and_generate

    # Preprocess
    t0 = time.perf_counter()
    prompt = build_prompt_for_task(query, task)
    batch = process_batch_and_generate(
        tokenizer,
        [(pil_image, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=args.min_dim,
        max_dimension=args.max_dim,
    )
    t_preprocess = time.perf_counter() - t0

    # Generate
    t0 = time.perf_counter()
    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        task=task,
    )
    t_generate = time.perf_counter() - t0

    # Decode + parse
    t0 = time.perf_counter()
    aux = aux_outputs[0]
    bboxes = pair_bbox_entries(aux.bboxes_raw)
    t_decode = time.perf_counter() - t0

    # Token counts
    all_toks = np.array(output_tokens[0]).flatten()
    n_prefill = batch["tokens"].shape[1]
    decoded_toks = all_toks[n_prefill:]
    eos_pos = np.where(
        (decoded_toks == tokenizer.eos_token_id) | (decoded_toks == tokenizer.pad_token_id)
    )[0]
    n_decoded = int(eos_pos[0] + 1) if len(eos_pos) > 0 else len(decoded_toks)

    return {
        "bboxes": bboxes,
        "masks_rle": aux.masks_rle,
        "n_prefill": n_prefill,
        "n_decoded": n_decoded,
        "t_preprocess": t_preprocess,
        "t_generate": t_generate,
        "t_decode": t_decode,
        "tok_per_sec": n_decoded / t_generate if t_generate > 0 else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Falcon-Perception MLX Demo")
    parser.add_argument("--image", type=str, default=None, help="Path or URL to image")
    parser.add_argument("--query", type=str, default=None, help="Comma-separated queries (default: cat)")
    parser.add_argument("--task", type=str, default="both", choices=["detection", "segmentation", "both"])
    parser.add_argument("--model-id", type=str, default=PERCEPTION_MODEL_ID)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--min-dim", type=int, default=256)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2, help="Warm inference runs to average")
    parser.add_argument("--out-dir", type=str, default="./output")
    args = parser.parse_args()

    queries = [q.strip() for q in (args.query or ",".join(DEFAULT_QUERIES)).split(",")]
    tasks = ["detection", "segmentation"] if args.task == "both" else [args.task]

    print("=" * 56)
    print("  Falcon-Perception MLX Demo")
    print("=" * 56)

    # ── Load model ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=args.model_id,
        dtype=args.dtype,
        backend="mlx",
    )
    t_model_load = time.perf_counter() - t0

    from falcon_perception.mlx.batch_inference import BatchInferenceEngine
    engine = BatchInferenceEngine(model, tokenizer)

    # ── Load image ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    image_src = args.image or DEFAULT_IMAGE
    pil_image = load_image(image_src).convert("RGB")
    t_image_load = time.perf_counter() - t0

    w, h = pil_image.size
    print(f"  Model  : {args.model_id}")
    print(f"  Dtype  : {args.dtype}")
    print(f"  Image  : {w} x {h}")
    print(f"  Queries: {queries}")
    print(f"  Tasks  : {tasks}")
    print()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run inference per task × query ────────────────────────────────
    all_results = []

    for task in tasks:
        for query in queries:
            label = f'{task} "{query}"'
            print(f"--- {label} ---")

            # Cold run
            x = 0
            for i in range(100):
                x = x + 1
                print(x)
                result = run_inference(engine, tokenizer, model_args, pil_image, query, task, args)
                n_obj = len(result["bboxes"])
                n_masks = len(result["masks_rle"])
                print(f"  Objects: {n_obj} bboxes, {n_masks} masks")
                for i, bbox in enumerate(result["bboxes"]):
                    mask_tag = "with mask" if i < n_masks else "no mask"
                    print(f"    [{i}] cx={bbox['x']:.3f} cy={bbox['y']:.3f} "
                          f"h={bbox['h']:.4f} w={bbox['w']:.4f}  ({mask_tag})")
                print(f"  Cold: {_fmt(result['t_generate'])} "
                      f"({result['n_decoded']} tokens, {result['tok_per_sec']:.1f} tok/s)")

            # Warm runs
            warm_times = []
            for _ in range(args.warmup):
                wr = run_inference(engine, tokenizer, model_args, pil_image, query, task, args)
                warm_times.append(wr["t_generate"])
            avg_warm = sum(warm_times) / len(warm_times) if warm_times else 0
            warm_tok_s = result["n_decoded"] / avg_warm if avg_warm > 0 else 0
            print(f"  Warm: {_fmt(avg_warm)} avg over {args.warmup} runs "
                  f"({warm_tok_s:.1f} tok/s)")

            # Visualize
            safe_q = "".join(c if c.isalnum() or c in " _-" else "_" for c in query)[:30].strip()
            stem = Path(args.image).stem if args.image else "demo"
            out_path = out_dir / f"{stem}_{safe_q}_{task}.jpg"
            visualize(pil_image, result["bboxes"], result["masks_rle"], str(out_path))
            print()

            all_results.append({
                "label": label,
                "cold": result,
                "warm_avg": avg_warm,
                "warm_tok_s": warm_tok_s,
            })

    # ── Timing summary ────────────────────────────────────────────────
    print("=" * 56)
    print("  Timing Summary")
    print("=" * 56)
    print(f"  Model loading ......... {_fmt(t_model_load):>10}")
    print(f"  Image loading ......... {_fmt(t_image_load):>10}")

    for r in all_results:
        c = r["cold"]
        print(f"  -- {r['label']} --")
        print(f"    Preprocess .......... {_fmt(c['t_preprocess']):>10}")
        print(f"    Generation (cold) ... {_fmt(c['t_generate']):>10}")
        print(f"    Generation (warm) ... {_fmt(r['warm_avg']):>10}")
        print(f"      Prefill tokens .... {c['n_prefill']:>10}")
        print(f"      Decoded tokens .... {c['n_decoded']:>10}")
        print(f"      Cold tok/s ........ {c['tok_per_sec']:>10.1f}")
        print(f"      Warm tok/s ........ {r['warm_tok_s']:>10.1f}")
        print(f"    Decode + parse ...... {_fmt(c['t_decode']):>10}")

    total = t_model_load + t_image_load + sum(
        r["cold"]["t_preprocess"] + r["cold"]["t_generate"] + r["cold"]["t_decode"] + r["warm_avg"] * args.warmup
        for r in all_results
    )
    print("-" * 56)
    print(f"  Total ................. {_fmt(total):>10}")
    print("=" * 56)
    print(f"\n  Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

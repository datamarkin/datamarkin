import time
import mlx.core as mx
import numpy as np
from PIL import Image
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


def visualize_semantic_mask(image: Image.Image, seg_mask: mx.array, alpha: float = 0.5,
                            color: tuple = (255, 0, 0)) -> Image.Image:
    """
    Overlay binary semantic segmentation mask on original image.

    Args:
        image: Original PIL image
        seg_mask: Semantic segmentation logits [B, 1, H, W] or [1, H, W]
        alpha: Transparency of the overlay (0-1)
        color: RGB color tuple for the mask overlay

    Returns:
        PIL Image with mask overlay
    """
    # Convert to numpy and apply sigmoid for probabilities
    seg_np = np.array(seg_mask)

    # Handle different shapes
    if seg_np.ndim == 4:
        seg_np = seg_np[0, 0]  # [B, C, H, W] -> [H, W]
    elif seg_np.ndim == 3:
        seg_np = seg_np[0]  # [C, H, W] -> [H, W]

    # Apply sigmoid to convert logits to probabilities
    seg_probs = 1 / (1 + np.exp(-seg_np))

    # Threshold to get binary mask
    seg_binary = (seg_probs > 0.5).astype(np.float32)

    # Resize mask to match image size if needed
    mask_h, mask_w = seg_binary.shape
    img_w, img_h = image.size

    if (mask_h, mask_w) != (img_h, img_w):
        mask_pil = Image.fromarray((seg_binary * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((img_w, img_h), Image.BILINEAR)
        seg_binary = np.array(mask_pil) / 255.0

    # Create colored overlay
    overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]  # R
    overlay[..., 1] = color[1]  # G
    overlay[..., 2] = color[2]  # B
    overlay[..., 3] = (seg_binary * alpha * 255).astype(np.uint8)  # Alpha

    # Composite overlay on image
    image_rgba = image.convert("RGBA")
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    result = Image.alpha_composite(image_rgba, overlay_img)

    return result.convert("RGB")


def save_semantic_mask(seg_mask: mx.array, output_path: str = "semantic_mask.png"):
    """
    Save the binary semantic mask as a grayscale image.

    Args:
        seg_mask: Semantic segmentation logits [B, 1, H, W]
        output_path: Path to save the mask
    """
    seg_np = np.array(seg_mask)

    # Handle shape
    if seg_np.ndim == 4:
        seg_np = seg_np[0, 0]
    elif seg_np.ndim == 3:
        seg_np = seg_np[0]

    # Sigmoid + threshold
    seg_probs = 1 / (1 + np.exp(-seg_np))
    seg_binary = (seg_probs > 0.5).astype(np.uint8) * 255

    mask_img = Image.fromarray(seg_binary, mode="L")
    mask_img.save(output_path)
    print(f"Semantic mask saved to: {output_path}")
    return mask_img


def main():
    start = time.perf_counter()
    model = build_sam3_image_model()

    second = time.perf_counter()
    print(f"Model loaded in {second - start:.2f} seconds.")

    image_path = "fietzfotos-road-7064492_1920.jpg"
    image = Image.open(image_path)
    width, height = image.size
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    inter = time.perf_counter()
    print(f"Image processed in {inter - second:.2f} seconds.")

    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="face")
    output = inference_state

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    third = time.perf_counter()
    print(f"Inference completed in {third - second:.2f} seconds.")
    print(f"Total Objects Found: {len(scores)}")
    print(f"Scores: {scores}")
    print(f"Boxes: {boxes}")

    # === Semantic Segmentation Visualization ===
    if "semantic_seg" in output:
        seg_mask = output["semantic_seg"]
        print(f"Semantic mask shape: {seg_mask.shape}")

        # Save the raw binary mask
        save_semantic_mask(seg_mask, "semantic_mask.png")

        # Create and save overlay visualization
        overlay_img = visualize_semantic_mask(
            image,
            seg_mask,
            alpha=0.5,
            color=(0, 255, 128)  # Green-ish overlay
        )
        overlay_img.save("semantic_overlay.png")
        print("Semantic overlay saved to: semantic_overlay.png")

        # Show the images (optional - comment out if running headless)
        # overlay_img.show()
    else:
        print("semantic_seg not found in output. Add this line in sam3_image_processor.py after line 187:")
        print('    state["semantic_seg"] = seg_mask')


if __name__ == "__main__":
    main()

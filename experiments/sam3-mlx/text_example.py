import time
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


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

    # Text prompts only: label "car" and "person"
    print("\n--- Text Prompts Only ---")

    # First prompt: "car"
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(
        state=inference_state, prompt="car"
    )

    masks, boxes, scores = (
        inference_state["masks"],
        inference_state["boxes"],
        inference_state["scores"],
    )

    t1 = time.perf_counter()
    print(f"\nText prompt 'car' inference in {t1 - inter:.2f} seconds.")
    print(f"Objects found: {len(scores)}")
    print(f"Scores: {scores}")
    print(f"Boxes: {boxes}")

    # Second prompt: "person"
    print("\n--- Prompting for 'person' ---")
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(
        state=inference_state, prompt="person"
    )

    masks, boxes, scores = (
        inference_state["masks"],
        inference_state["boxes"],
        inference_state["scores"],
    )

    t2 = time.perf_counter()
    print(f"\nText prompt 'person' inference in {t2 - inter:.2f} seconds.")
    print(f"Objects found: {len(scores)}")
    print(f"Scores: {scores}")
    print(f"Boxes: {boxes}")


if __name__ == "__main__":
    main()

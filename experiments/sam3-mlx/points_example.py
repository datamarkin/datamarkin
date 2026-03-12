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

    # Point prompts only: 3 positive + 2 negative points
    print("\n--- Point Prompts Only ---")
    processor.reset_all_prompts(inference_state)

    # Normalize coordinates to [0, 1]
    positive_points = [
        [1473 / width, 810 / height],   # center of road surface
        [1475 / width, 863 / height],   # lower road
        [1480 / width, 900 / height],   # another point on road
    ]
    negative_points = [
        [1487 / width, 877 / height],   # not this region (sky/building)
        [500 / width, 200 / height],    # not this region (background)
    ]

    all_points = positive_points + negative_points
    all_labels = [True] * len(positive_points) + [False] * len(negative_points)

    inference_state = processor.add_points_prompt(
        points=all_points,
        labels=all_labels,
        state=inference_state,
    )

    masks, boxes, scores = (
        inference_state["masks"],
        inference_state["boxes"],
        inference_state["scores"],
    )

    t1 = time.perf_counter()
    print(f"Point prompt inference in {t1 - inter:.2f} seconds.")
    print(f"Objects found: {len(scores)}")
    print(f"Scores: {scores}")
    print(f"Boxes: {boxes}")


if __name__ == "__main__":
    main()

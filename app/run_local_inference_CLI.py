import argparse
import json
from pathlib import Path

from app.inference_service import InferenceService


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run industrial panel object detection + OCR inference."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--output-folder",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "crops"),
        help="Folder to save crops and visualization.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "panel_data.json"),
        help="Path to save JSON output.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Detection score threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_path = Path(args.image)
    output_folder = Path(args.output_folder)
    json_output = Path(args.json_output)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    service = InferenceService(score_thresh=args.threshold)
    result = service.predict(
        image_path=image_path,
        output_folder=output_folder,
        save_visualization=True,
        save_crops=True,
    )

    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(result["detections"], f, indent=4)

    print("Done.")
    print(f"JSON saved to: {json_output.resolve()}")
    print(f"Crops saved to: {output_folder.resolve()}")
    if result["visualization_path"]:
        print(f"Visualization saved to: {result['visualization_path']}")
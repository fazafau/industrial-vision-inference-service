import os
import cv2
import torch
import json
import numpy as np
import re
import argparse
from pathlib import Path

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from paddleocr import PaddleOCR


# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "config.yaml"
DEFAULT_MODEL_WEIGHTS_PATH = PROJECT_ROOT / "models" / "model_final_clean.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


# =========================================================
# CLASS MAP
# =========================================================
CLASS_MAP = {
    0: "1_pole_circuit_breaker",
    1: "1_pole_installation_contactor",
    2: "2_pole_circuit_breaker",
    3: "2_pole_installation_contactor",
    4: "2_pole_RCBO_overcurrent",
    5: "2_pole_RCCB",
    6: "2_pole_switch_disconnector",
    7: "3_pole_circuit_breaker",
    8: "4_pole_circuit_breaker",
    9: "4_pole_installation_contactor",
    10: "4_pole_RCBO_overcurrent",
    11: "4_pole_RCCB",
    12: "4_pole_SPD",
    13: "4_pole_switch_disconnector",
    14: "DIN_rails",
    15: "bell_transformer",
    16: "indicator_pilot_light",
    17: "phase_monitoring_relay",
    18: "relay",
    19: "rotary_cam_switches",
    20: "schuko_socket_outlet",
    21: "screw_type_fuse",
    22: "surge_protection_module",
    23: "terminal_blocks",
    24: "twilight_switch"
}


# =========================================================
# TEXT EXTRACTION
# =========================================================
def extract_component_info(raw_text: str):
    clean_text = raw_text.upper()
    clean_text = re.sub(r'[^A-Z0-9/\.\-\s]', '', clean_text)

    info = {"brand": "Unknown_Brand", "model": "", "full_label": ""}

    if "SCHNEIDER" in clean_text:
        info["brand"] = "Schneider Electric"
        clean_text = clean_text.replace("IC6ON", "IC60N").replace("IC4ON", "IC40N")

        if "IC60N" in clean_text:
            info["model"] = "IC60N"
        elif "IC40N" in clean_text:
            info["model"] = "IC40N"
        elif "ACTI9" in clean_text:
            info["model"] = "Acti9"
        elif "RES19" in clean_text or "RESI9" in clean_text:
            info["model"] = "Resi9"

        match = re.search(r'\b([BCD][0-9]{1,3})A?\b', clean_text)
        if match:
            rating = match.group(1)
            if info["model"]:
                info["model"] += " " + rating
            else:
                info["model"] = rating

        if "ICT" in clean_text:
            info["model"] = "ICT"
        if "ILD" in clean_text:
            info["model"] = "iID"
        if "RCBO" in clean_text:
            info["model"] = "RCBO"
        if "IDPN" in clean_text:
            info["model"] = "iDPN N Vigi"
        if "TR" in clean_text and "RESI9" in clean_text:
            info["model"] = "Resi9 TR"
        if "IPRD1" in clean_text:
            info["model"] = "iPRD1 (SPD)"
        if "ISSW" in clean_text:
            info["model"] = "iSSW"
        elif "ISW" in clean_text:
            info["model"] = "iSW"

    elif "ABB" in clean_text or "ESB25" in clean_text:
        info["brand"] = "ABB"
        clean_text = clean_text.replace("ES8", "ESB")

        if "E251" in clean_text:
            info["model"] = "E251"
        elif "E252" in clean_text:
            info["model"] = "E252"
        elif "ESB" in clean_text:
            match = re.search(r'(ESB\s?[0-9]+-?[0-9N]*)', clean_text)
            if match:
                info["model"] = match.group(1)
        match = re.search(r'(S\s?2[0-9]{2}\s?[A-Z0-9]*)', clean_text)
        if match:
            info["model"] = match.group(1).replace(" ", "")

    elif "SIEMENS" in clean_text:
        info["brand"] = "Siemens"
        if "5SV" in clean_text:
            match = re.search(r'(5SV[0-9]\s?[0-9]+-?[0-9A-Z]*)', clean_text)
            if match:
                info["model"] = match.group(1).replace(" ", "")

    elif "HAGER" in clean_text:
        info["brand"] = "Hager"
        match = re.search(r'(MBN\s?[0-9]+)', clean_text)
        if match:
            info["model"] = match.group(1)
        match = re.search(r'((E|ES|XE)\s?[0-9]{3}[A-Z0-9]*)', clean_text)
        if match:
            info["model"] = match.group(1)
        if "CGA" in clean_text:
            match = re.search(r'(CGA\s?[0-9]{3}+[A-Z]?)', clean_text)
            if match:
                info["model"] = match.group(1)

    if info["brand"] != "Unknown_Brand":
        info["model"] = info["model"].strip()
        if info["model"]:
            info["full_label"] = f"{info['brand']} {info['model']}"
        else:
            info["full_label"] = info["brand"]
    else:
        info["full_label"] = None

    return info["full_label"]


# =========================================================
# JSON ENCODER
# =========================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =========================================================
# MODEL LOADING
# =========================================================
def load_d2_checkpoint_trusted(model, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpointer = DetectionCheckpointer(model)
    checkpointer._load_model(checkpoint)


def load_models(
    model_config_path: Path,
    model_weights_path: Path,
    score_thresh: float = 0.90,
):
    print(f"Loading Detectron2 config from: {model_config_path}")
    cfg = LazyConfig.load(str(model_config_path))

    det_model = instantiate(cfg.model)
    det_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_model.to(device)

    print(f"Loading weights from: {model_weights_path}")
    load_d2_checkpoint_trusted(det_model, str(model_weights_path))

    det_model.roi_heads.box_predictor.test_score_thresh = score_thresh

    print("Loading PaddleOCR...")
    ocr_model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    return det_model, ocr_model, device


# =========================================================
# PIPELINE
# =========================================================
def process_panel_image(
    image_path: Path,
    output_folder: Path,
    json_output: Path,
    model_config_path: Path,
    model_weights_path: Path,
    score_thresh: float = 0.90,
):
    det_model, ocr_model, device = load_models(
        model_config_path=model_config_path,
        model_weights_path=model_weights_path,
        score_thresh=score_thresh,
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original_height, original_width = img_bgr.shape[:2]
    img_rgb = img_bgr[:, :, ::-1].copy()

    aug = T.ResizeShortestEdge(short_edge_length=640, max_size=640, sample_style="choice")
    transform = aug.get_transform(img_rgb)
    img_resized = transform.apply_image(img_rgb)
    tensor_input = torch.as_tensor(img_resized.transpose(2, 0, 1)).float().to(device)

    inputs = [{
        "image": tensor_input,
        "height": original_height,
        "width": original_width
    }]

    print("Running detection...")
    with torch.no_grad():
        outputs = det_model(inputs)[0]

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    max_id = max(CLASS_MAP.keys())
    thing_classes = [CLASS_MAP.get(i, f"Class_{i}") for i in range(max_id + 1)]

    viz_meta_name = "temp_viz_metadata"
    try:
        MetadataCatalog.remove(viz_meta_name)
    except KeyError:
        pass

    MetadataCatalog.get(viz_meta_name).set(thing_classes=thing_classes)
    viz_metadata = MetadataCatalog.get(viz_meta_name)

    vis = Visualizer(img_rgb, metadata=viz_metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    vis_output = vis.draw_instance_predictions(instances)
    vis_img = vis_output.get_image()[:, :, ::-1]

    vis_save_path = output_folder / "visualized_detection.jpg"
    cv2.imwrite(str(vis_save_path), vis_img)

    base_filename = image_path.stem
    detected_components = []

    print(f"Found {len(boxes)} objects. Running OCR...")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_width, x2), min(original_height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop_img = img_bgr[y1:y2, x1:x2]

        class_id = int(classes[i])
        score = float(scores[i])

        save_name = f"{base_filename}_cls{class_id}_conf{score:.2f}_id{i}.jpg"
        save_path = output_folder / save_name
        cv2.imwrite(str(save_path), crop_img)

        ocr_result = ocr_model.ocr(crop_img, cls=True)
        extracted_texts = []

        if ocr_result and ocr_result[0] is not None:
            for line in ocr_result[0]:
                text_content = line[1][0]
                text_conf = line[1][1]
                if text_conf > 0.6:
                    extracted_texts.append(text_content)

        full_text = " ".join(extracted_texts)
        clean_label = extract_component_info(full_text)
        class_name = CLASS_MAP.get(class_id, f"Unknown_Class_{class_id}")

        component_data = {
            "id": i,
            "class_id": class_id,
            "class_name": class_name,
            "confidence_score": score,
            "bbox": [x1, y1, x2, y2],
            "crop_filename": save_name,
            "raw_text": full_text,
            "label": clean_label,
        }

        detected_components.append(component_data)

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(detected_components, f, cls=NumpyEncoder, indent=4)

    print("Done.")
    print(f"Visualization saved to: {vis_save_path.resolve()}")
    print(f"JSON saved to: {json_output.resolve()}")
    print(f"Crops saved to: {output_folder.resolve()}")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run industrial panel object detection + OCR inference."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image."
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "crops"),
        help="Folder to save crops and visualization."
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "panel_data.json"),
        help="Path to save JSON output."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Detection score threshold."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG_PATH),
        help="Path to Detectron2 config.yaml."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_MODEL_WEIGHTS_PATH),
        help="Path to model checkpoint."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_path = Path(args.image)
    output_folder = Path(args.output_folder)
    json_output = Path(args.json_output)
    model_config_path = Path(args.config)
    model_weights_path = Path(args.weights)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {model_config_path}")
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {model_weights_path}")

    process_panel_image(
        image_path=image_path,
        output_folder=output_folder,
        json_output=json_output,
        model_config_path=model_config_path,
        model_weights_path=model_weights_path,
        score_thresh=args.threshold,
    )
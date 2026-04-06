````markdown
# Industrial Vision Inference Service

A deployment-oriented computer vision inference pipeline for industrial control-cabinet images, built from my master’s thesis work.

This project loads a trained Detectron2 object detection model, runs inference on industrial panel images, applies OCR on detected components, and generates structured outputs including predicted classes, bounding boxes, confidence scores, OCR text, and extracted labels.

## Features

- Object detection on industrial electrical/control-cabinet images
- OCR on detected components using PaddleOCR
- Structured JSON output for downstream use
- Automatic saving of cropped detections
- Annotated visualization of detections
- CLI-based execution for flexible local inference

## Project Structure

```text
industrial vision inference service/
│
├── app/
│   └── run_local_inference.py
├── models/
│   ├── config.yaml
│   └── model_final_clean.pth
├── sample_inputs/
├── outputs/
├── requirements.txt
├── requirements_locked.txt
├── .gitignore
└── README.md
````

## Requirements

This project was tested in a Python 3.11 virtual environment on Windows.

Main dependencies include:

* PyTorch
* Detectron2
* PaddleOCR
* PaddlePaddle
* timm
* fairscale
* OpenCV

## Setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venvironment
.venvironment\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If needed, install Detectron2 separately depending on your local environment.

## Usage

Run inference on an input image:

```powershell
python app/run_local_inference.py --image sample_inputs/sample6.png
```

Run with a custom output folder and JSON file:

```powershell
python app/run_local_inference.py --image sample_inputs/sample6.png --output-folder outputs/test_run --json-output outputs/test_run/results.json
```

Run with a custom detection threshold:

```powershell
python app/run_local_inference.py --image sample_inputs/sample6.png --threshold 0.85
```

## Output

The pipeline generates:

* Cropped detections
* An annotated visualization image
* A JSON file containing structured inference results

Each detected object includes:

* `id`
* `class_id`
* `class_name`
* `confidence_score`
* `bbox`
* `crop_filename`
* `raw_text`
* `label`

## Current Status

The local CLI pipeline is working end-to-end:

* model loading
* detection
* OCR
* visualization
* JSON export

Next planned steps:

* FastAPI wrapper
* Dockerization
* deployment-oriented project cleanup

## Author

Fazal Kadri

```
```

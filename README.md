
# Industrial Vision Inference Service

A deployment-oriented computer vision inference pipeline for industrial control-cabinet images, built from my master’s thesis work.

This project loads a trained Detectron2 object detection model, runs inference on industrial panel images, applies OCR on detected components, and exposes the pipeline through both a local CLI workflow and a FastAPI-based backend service. The service can return structured JSON predictions as well as annotated visualization images, and the full application has been containerized with Docker for reproducible local deployment.

```mermaid
flowchart TD
    A[Input Image] --> B[CLI or FastAPI Request]

    B --> C[Load Detectron2 Config]
    C --> D[Load Trained Checkpoint]
    D --> E[Preprocess Image]
    E --> F[Object Detection Inference]

    F --> G[Detected Bounding Boxes and Classes]
    G --> H[Crop Detected Components]
    H --> I[Run PaddleOCR on Crops]
    I --> J[Regex-Based Label Extraction]

    J --> K[Structured Output Assembly]
    K --> L[JSON Response]
    K --> M[Annotated Visualization Image]
    K --> N[Saved Detection Crops]

    subgraph Interfaces
        B1[CLI: run_local_inference.py]
        B2[FastAPI: /predict]
        B3[FastAPI: /predict-visualized]
    end

    B1 --> B
    B2 --> B
    B3 --> B

    subgraph Deployment
        O[FastAPI App]
        P[Docker Container]
    end

    O --> P
    B2 --> O
    B3 --> O
```
## Features

- Object detection on industrial electrical/control-cabinet images
- OCR on detected components using PaddleOCR
- Structured JSON output for downstream use
- Automatic saving of cropped detections
- Annotated visualization of detections
- CLI-based execution for flexible local inference
- FastAPI backend for HTTP-based inference
- Interactive API testing through Swagger UI (`/docs`)
- Endpoint for returning visualization images directly
- Dockerized deployment for reproducible local execution

## Project Structure

```text
industrial vision inference service/
│
├── app/
│   ├── inference_service.py
│   ├── main.py
│   └── run_local_inference.py
├── models/
│   ├── config.yaml
│   └── model_final_clean.pth
├── sample_inputs/
├── outputs/
├── requirements.txt
├── requirements_docker.txt
├── requirements_locked.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md


## Requirements

This project was developed and tested in a Python 3.11 environment.

Main dependencies include:

* PyTorch
* Detectron2
* PaddleOCR
* PaddlePaddle
* FastAPI
* Uvicorn
* timm
* fairscale
* OpenCV

## Local Setup

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

## CLI Usage

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

## FastAPI Usage

Run the API locally:

```powershell
uvicorn app.main:app --reload
```

Open the interactive API documentation in your browser:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

* `GET /`
  Root endpoint for basic service response or redirect target
* `GET /health`
  Health check endpoint
* `GET /model-info`
  Returns basic information about the deployed model
* `POST /predict`
  Accepts an uploaded image and returns structured JSON predictions
* `POST /predict-visualized`
  Accepts an uploaded image and returns the annotated visualization image

## Docker Usage

Build the Docker image:

```powershell
docker build -t industrial-vision-api .
```

Run the container:

```powershell
docker run -p 8000:8000 industrial-vision-api
```

Then access the API at:

```text
http://127.0.0.1:8000/docs
```

## Output

The pipeline can generate:

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

The project currently supports:

* local end-to-end inference
* model loading
* object detection
* OCR
* visualization generation
* JSON export
* CLI execution
* FastAPI-based inference service
* image upload through HTTP endpoints
* annotated image response through API
* Dockerized local deployment

## Notes

* A separate `requirements_docker.txt` file is used for Docker builds to avoid platform-specific local packages.
* The Docker image is CPU-oriented for local reproducibility and to avoid unnecessary GPU/CUDA dependencies in the containerized setup.
* The model weights are copied into the container during build so the service can run independently of the local host environment.

## Author

Fazal Kadri


# AGENTS.md

## Purpose

This repository is a small industrial vision inference service built around one shared Python inference pipeline.

The codebase has two user-facing entry points:

- FastAPI service in `app/main.py`
- Local CLI runner in `app/run_local_inference_CLI.py`

Both call the same core class:

- `InferenceService` in `app/inference_service.py`


## Project Structure

Top-level files and folders that matter for day-to-day work:

- `app/main.py`
  FastAPI app, API routes, temp-file upload handling, and the singleton `InferenceService`
- `app/inference_service.py`
  Core runtime pipeline: model loading, image preprocessing, Detectron2 inference, OCR, regex label extraction, visualization, and response assembly
- `app/run_local_inference_CLI.py`
  Local command-line runner for inference on a single image
- `models/config.yaml`
  Detectron2 lazy config used to instantiate the model
- `models/model_final_clean.pth`
  Trained model weights used at inference time
- `README.md`
  User-facing setup and usage notes
- `Dockerfile`
  CPU-oriented container build for the FastAPI service
- `requirements.txt`
  Main local dependency set
- `requirements_docker.txt`
  Slimmer Docker dependency set
- `sample_inputs/`
  Example images for local testing
- `outputs/`
  Default output location for CLI runs


## Architecture

The service is intentionally simple and centralized.

Request flow:

1. API or CLI receives an image path or uploaded file.
2. `InferenceService.load_models()` lazily loads the Detectron2 model and PaddleOCR.
3. `InferenceService.predict()` reads the image with OpenCV.
4. The image is resized and passed into the Detectron2 model.
5. Each detected bounding box is cropped.
6. PaddleOCR runs on each crop.
7. OCR text is post-processed with regex heuristics in `extract_component_info()`.
8. The service returns structured JSON and may also save crops or a visualization image.

Important constraint:

- Most business logic lives in `app/inference_service.py`. Changes there affect both the API and CLI.


## Main Commands

Use the commands below as the source of truth for this repo’s current structure.

### Environment setup

```powershell
py -3.11 -m venv .venvironment
.venvironment\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the API locally

```powershell
uvicorn app.main:app --reload
```

Docs will be available at:

```text
http://127.0.0.1:8000/docs
```

### Run local CLI inference

Actual CLI file in this repo:

```powershell
python app/run_local_inference_CLI.py --image sample_inputs/sample6.png
```

With custom output paths:

```powershell
python app/run_local_inference_CLI.py --image sample_inputs/sample6.png --output-folder outputs/test_run --json-output outputs/test_run/results.json
```

With a custom threshold:

```powershell
python app/run_local_inference_CLI.py --image sample_inputs/sample6.png --threshold 0.85
```

### Build and run Docker image

```powershell
docker build -t industrial-vision-api .
docker run -p 8000:8000 industrial-vision-api
```


## API Endpoints

Defined in `app/main.py`:

- `GET /`
  Redirects to `/docs`
- `GET /health`
  Lightweight health response
- `GET /model-info`
  Static model metadata response
- `POST /predict`
  Returns structured JSON detections
- `POST /predict-visualized`
  Returns the generated visualization image bytes


## Coding Patterns In Use

Match the patterns already present unless there is a good reason to refactor.

- Use `pathlib.Path` for repo-relative paths and filesystem work.
- Keep the FastAPI layer thin and push inference behavior into `InferenceService`.
- Reuse the existing singleton-style `InferenceService` pattern in `app/main.py` unless deliberately changing lifecycle behavior.
- Prefer explicit booleans for output behavior such as `save_visualization` and `save_crops`.
- Keep detection response objects as plain Python dicts/lists with JSON-serializable primitives.
- Follow the current import style: standard library, third-party imports, then local imports.
- Preserve the current OpenCV and Detectron2 image flow unless intentionally updating preprocessing.
- Preserve the `CLASS_MAP` convention when adding or renaming classes.
- Keep OCR post-processing logic near inference logic unless doing a deliberate extraction/refactor.


## Repo-Specific Gotchas

- The README refers to `app/run_local_inference.py`, but the actual file in the repo is `app/run_local_inference_CLI.py`.
- `models/model_final_clean.pth` is large and required for real inference.
- `models/config.yaml` includes training-era fields and absolute Linux paths, but inference currently uses the model portion through `LazyConfig.load(...)`.
- The FastAPI handlers are `async`, but the work done inside them is blocking and compute-heavy.
- `InferenceService.load_models()` is lazy, so the first request or first CLI run pays the model startup cost.
- There is no test suite in the repository right now.


## When Editing

- If you change output schema in `InferenceService.predict()`, update both API and CLI expectations.
- If you change model-loading behavior, verify both local CLI usage and `uvicorn app.main:app`.
- If you rename CLI flags or paths, update `README.md` at the same time.
- If you change dependencies, check whether both `requirements.txt` and `requirements_docker.txt` need corresponding updates.
- If you change anything in `models/config.yaml` assumptions, make sure the runtime still works with `LazyConfig.load(...)`.


## Verification

There are no automated tests checked in, so prefer lightweight manual verification:

1. Start the API with `uvicorn app.main:app --reload`.
2. Hit `/health` and `/docs`.
3. Run one local CLI inference against a sample image.
4. If you changed visualization or output behavior, verify both `/predict` and `/predict-visualized`.


## Good First Refactor Targets

If the repo is being improved rather than just maintained, the highest-value cleanup areas are:

- extracting OCR label parsing from `app/inference_service.py`
- separating response shaping from model execution
- adding smoke tests for CLI and API startup
- improving model lifecycle and startup/health reporting

## Test
- pytest -q

## Lint/format
- ruff check .
- black .

## Rules
- Do not change API response schemas unless asked
- Keep functions small
- Prefer typed functions
- Add tests for bug fixes
- Explain breaking changes before making them

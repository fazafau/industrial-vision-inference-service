from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response

from app.inference_service import InferenceService

app = FastAPI(title="Industrial Vision Inference Service")

service = InferenceService(score_thresh=0.90)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": "Detectron2 industrial vision model",
        "num_classes": 25,
        "ocr_enabled": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".png"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        temp_image_path = tmpdir_path / f"input{suffix}"
        output_folder = tmpdir_path / "outputs"

        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = service.predict(
                image_path=temp_image_path,
                output_folder=output_folder,
                save_visualization=False,
                save_crops=False,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-visualized")
async def predict_visualized(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".png"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        temp_image_path = tmpdir_path / f"input{suffix}"
        output_folder = tmpdir_path / "outputs"

        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = service.predict(
                image_path=temp_image_path,
                output_folder=output_folder,
                save_visualization=True,
                save_crops=False,
            )

            vis_path = result["visualization_path"]
            if not vis_path or not Path(vis_path).exists():
                raise HTTPException(status_code=500, detail="Visualization image not created.")

            with open(vis_path, "rb") as f:
                image_bytes = f.read()

            return Response(content=image_bytes, media_type="image/jpeg")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
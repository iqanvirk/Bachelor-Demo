import base64
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from use_model import analyze_image

app = FastAPI(title="OCT Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_for_json(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": result.get("prediction"),
        "final_prediction": result.get("final_prediction"),
        "image_prediction": result.get("image_prediction"),
        "thickness_prediction": result.get("thickness_prediction"),
        "confidence": result.get("confidence"),
        "class_probabilities": result.get("class_probabilities"),
        "image_model_probabilities": result.get("image_model_probabilities"),
        "thickness_probabilities": result.get("thickness_probabilities"),
        "backend": result.get("backend"),
        "layer_metrics": result.get("layer_metrics"),
        "total_retinal_metrics": result.get("total_retinal_metrics"),
        "pathology_score": result.get("pathology_score"),
        "overlay_base64": pil_to_base64(result["overlay"]) if result.get("overlay") is not None else None,
        "colored_mask_base64": pil_to_base64(result["colored_mask"]) if result.get("colored_mask") is not None else None,
        "mask_with_legend_base64": pil_to_base64(result["mask_with_legend"]) if result.get("mask_with_legend") is not None else None,
    }


@app.get("/")
def root():
    return {"message": "OCT Analysis API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    file_like = BytesIO(image_bytes)
    result = analyze_image(file_like)
    return normalize_for_json(result)
from fastapi import FastAPI, File, UploadFile, Query

from app.predictor import predictor

app = FastAPI(title="Dog Vision API", version="2.0.0")


@app.get("/")
def home():
    return {
        "message": "Dog Vision API",
        "version": "2.0.0",
        "endpoints": {
            "POST /predict": "Upload an image for breed prediction",
            "GET /breeds": "List all supported breeds",
        },
    }


@app.get("/breeds")
def list_breeds():
    """Return all 120 supported dog breeds."""
    return {
        "total": len(predictor.labels),
        "breeds": predictor.labels,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tta: bool = Query(False, description="Enable test-time augmentation for better accuracy"),
):
    """
    Predict the dog breed from an uploaded image.

    - **file**: JPEG or PNG image of a dog
    - **tta**: Set to true to enable test-time augmentation (slower but more accurate)
    """
    image_bytes = await file.read()
    result = predictor.predict(image_bytes, use_tta=tta)
    return result
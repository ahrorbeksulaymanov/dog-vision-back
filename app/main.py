from fastapi import FastAPI, File, UploadFile

from app.predictor import predictor

app = FastAPI(title="Dog Vision API")


@app.get("/")
def home():
    return {
        "message": "Dog Vision API"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    result = predictor.predict(image_bytes)

    return result
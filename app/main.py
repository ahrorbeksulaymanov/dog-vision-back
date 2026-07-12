import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.predictor import predictor
from app.pipeline import pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.detector import detector

    try:
        _ = detector.model
        print("[startup] Dog detector loaded (EfficientDet-Lite0)")
    except Exception as e:
        print(f"[startup] Warning: detector preload failed ({e})")
    yield


app = FastAPI(title="Dog Vision API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "message": "Dog Vision API",
        "version": "3.0.0",
        "endpoints": {
            "GET /": "PWA frontend (camera + live detection)",
            "GET /breeds": "List all 120 supported breeds",
            "POST /predict": "Upload an image for breed prediction",
            "WS /ws/live": "WebSocket for live camera detection",
        },
    }


@app.get("/breeds")
def list_breeds():
    return {
        "total": len(predictor.labels),
        "breeds": predictor.labels,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tta: bool = Query(False, description="Enable test-time augmentation"),
):
    image_bytes = await file.read()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predictor.predict, image_bytes, tta)


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await ws.receive_bytes()
            result = await loop.run_in_executor(None, pipeline.process_frame, data)
            await ws.send_json(result)
    except Exception:
        pass


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
# Dog Vision — Real-Time Dog Breed Detection

Point your phone camera at a dog and see its breed in real time. Supports **120 breeds** with bounding boxes around each dog in the frame. Multiple dogs detected simultaneously.

**Breed classifier accuracy: 92.42% top-1 | 99.30% top-3** (ensemble of EfficientNetV2S + ConvNeXtTiny)

---

## How It Works

```
Phone Camera → WebSocket frame → Dog Detector (EfficientDet-Lite0)
                                     ↓ bounding boxes
                                  Breed Classifier (your ensemble)
                                     ↓ breed per dog
                                  Return: [{bbox, breed, confidence}]
```

1. **Dog detection** uses a pre-trained EfficientDet-Lite0 (5MB) from TensorFlow Hub — finds all dogs in frame and draws bounding boxes.
2. **Breed classification** uses your existing ensemble (EfficientNetV2S + ConvNeXtTiny) — classifies each detected dog into one of 120 breeds.

---

## Environment

- **Python** 3.11
- **FastAPI** 0.139.0
- **TensorFlow** 2.21.0
- **TensorFlow Hub** 0.16.1

## Project Structure

```
app/
├── main.py            # FastAPI app (API endpoints, WebSocket, static files)
├── predictor.py       # DogBreedPredictor — ensemble, TTA, temperature scaling
├── preprocessing.py   # Image preprocessing (224x224, TTA variants)
├── detector.py        # DogDetector — EfficientDet-Lite0 from TF Hub
├── pipeline.py        # LivePipeline — detection + classification combined
└── static/
    ├── index.html     # PWA frontend (camera, canvas, WebSocket client)
    ├── manifest.json  # PWA manifest
    └── sw.js          # Service worker
data/
└── unique_breeds.json # 120 breed labels
models/
├── dog_model_improved.keras          # EfficientNetV2S
├── dog_model_convnext.keras          # ConvNeXtTiny
├── *_temp_scale.json                 # per-model confidence calibration
└── archive/dog_model.h5              # old MobileNetV2 (kept for reference)
Dockerfile             # Hugging Face Spaces deployment
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | PWA frontend (camera + live detection) |
| GET | `/api/health` | Health check |
| GET | `/breeds` | List all 120 supported breeds |
| POST | `/predict` | Upload an image, returns breed prediction |
| POST | `/predict?tta=true` | Prediction with test-time augmentation |
| WS | `/ws/live` | WebSocket — live camera frame processing |

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download the models

```bash
gh release download v2.0-ensemble --dir models
# or download manually from the Releases page into models/
```

## Running Locally

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` on your phone or desktop. Allow camera access.

## Deploy to Hugging Face Spaces (Free)

### 1. Create a Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and create a **Docker** space (choose "Docker" as the SDK, "Blank" template). Add this metadata to the top of the Space's `README.md`:

```yaml
---
title: Dog Vision
emoji: 🐕
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

### 2. Push your code

```bash
# Clone the space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/dog-vision
cd dog-vision

# Copy all project files (including models/)
cp -r /path/to/dog-vision-back/* .

# Models are gitignored — add them with Git LFS (HF Spaces supports this)
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/*.keras models/*.json data/*.json app/ Dockerfile .dockerignore Dockerfile requirements.txt

# Commit and push
git commit -m "Deploy Dog Vision to HF Spaces"
git push
```

### 3. Wait for build

HF Spaces will build the Docker image (first build takes ~10-15 minutes — TensorFlow is large). The detector model (EfficientDet-Lite0, ~6MB) is cached during the build.

### 4. Open on your phone

Visit `https://YOUR_USERNAME-dog-vision.hf.space` in your phone browser. Allow camera and tap the red button.

### Configuration

The space needs at least **2GB RAM** (TensorFlow + models). Free HF Spaces provide 16GB RAM by default — no changes needed.

## Single Image Prediction (API)

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@dog.jpg"
```

```json
{
    "primary": { "breed": "golden_retriever", "confidence": 0.95 },
    "top_k": [
        {"rank": 1, "breed": "golden_retriever", "confidence": 0.95}
    ],
    "is_unknown": false,
    "message": "Prediction successful."
}
```

## Live Detection Response (WebSocket)

Server returns per frame:

```json
{
  "detections": [
    {
      "bbox": [120, 80, 380, 420],
      "breed": "golden_retriever",
      "breed_conf": 0.92,
      "det_conf": 0.87
    }
  ],
  "fps": 4.5,
  "width": 640,
  "height": 480
}
```

## Model Architecture

Each classifier shares the same head; only the backbone differs:

```
Input (224×224×3)
  → Rescaling(255)
  → Backbone (EfficientNetV2S or ConvNeXtTiny), ImageNet-pretrained
  → GlobalAveragePooling2D
  → Dropout(0.4)
  → Dense(512, relu)
  → Dropout(0.2)
  → Dense(120, softmax)
```

The API averages both models' temperature-calibrated softmax outputs.

---

Built by **Mozzam**.

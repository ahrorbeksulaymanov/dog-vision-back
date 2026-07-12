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

## Deploy to Hugging Face Spaces

**Cost: ~$0.03/hour** (8 vCPU, 32GB RAM) with 15-min auto-sleep. At ~1 hour/day usage = **~$1/month**.

### Quick Deploy (automated)

```bash
# From the project root:
./deploy_hfspaces.sh
```

This script clones your HF Space, copies all files, sets up Git LFS, and pushes.

### Manual Deploy

#### 1. Create a Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space):
- **Space name**: `dog-vision`
- **SDK**: Docker (Blank)
- **Hardware**: 8 vCPU / 32GB RAM (upgrade from free tier after creating)
- **Sleep settings**: 15 min inactivity timeout

#### 2. Push your code

```bash
git clone https://huggingface.co/spaces/mozzamshahid/dog-vision
cd dog-vision

# Copy all project files
cp -r /path/to/dog-vision-back/app .
cp -r /path/to/dog-vision-back/models .
cp -r /path/to/dog-vision-back/data .
cp /path/to/dog-vision-back/{Dockerfile,.dockerignore,requirements.txt} .

# Track model weights with Git LFS (they're too big for regular git)
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add -A

# Create HF metadata README
cat > README.md << 'EOF'
---
title: Dog Vision
emoji: 🐕
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
Dog Vision — Live dog breed detection from your phone camera.
EOF

git commit -m "Deploy Dog Vision to HF Spaces"
git push
```

#### 3. Set the hardware

After the Space is created, go to **Settings → Hardware** and select **8 vCPU / 32GB RAM**.

#### 4. Wait for build (~10-15 min)

The first Docker build is slow because TensorFlow is large (~500MB pip package). The EfficientDet-Lite0 detector (~6MB) downloads during the build and is cached.

#### 5. Open on your phone

Visit `https://mozZamshahid-dog-vision.hf.space` on your phone:
1. Allow camera access
2. Tap the **red button** to start
3. Point at a dog — colored bounding boxes with breed names appear in real time

### Cost Breakdown

| Setting | Value |
|---------|-------|
| Hardware | 8 vCPU + 32GB RAM |
| Price | $0.03/hour |
| Auto-sleep | After 15 min inactivity |
| Cold start | ~30-60 sec (models load into memory) |
| Usage: 1 hr/day | ~$0.90/month |
| Usage: 2 hr/day | ~$1.80/month |

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

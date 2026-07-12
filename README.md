# Dog Vision — Real-Time Dog Breed Detection

Point your phone camera at a dog and see its breed in real time. Supports **120 breeds** with colored bounding boxes around each dog. Multiple dogs detected and classified simultaneously. Works as a **PWA** — open in your mobile browser, no app install needed.

**Accuracy: 92.42% top-1 | 99.30% top-3** (2-model ensemble of EfficientNetV2S + ConvNeXtTiny)

---

## Quick Start (5 minutes)

### Prerequisites

- **Python 3.11** (required — check with `python3.11 --version`)
- The trained model files (see step 2 below)

### 1. Clone and install

```bash
git clone https://github.com/MozzamShahid/dog-vision-back.git
cd dog-vision-back
python3.11 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the trained models

The model weights are published as a GitHub Release (too large for git). Download both `.keras` files into the `models/` folder:

```bash
# Option A — using GitHub CLI
gh release download v2.0-ensemble --dir models

# Option B — download manually from:
# https://github.com/MozzamShahid/dog-vision-back/releases/tag/v2.0-ensemble
# Place both .keras files in:   models/
```

After downloading, `models/` should contain:

```
models/
├── dog_model_improved.keras       # EfficientNetV2S (21M params)
├── dog_model_convnext.keras       # ConvNeXtTiny (28M params)
├── dog_model_improved_temp_scale.json   # calibration T=0.67
├── dog_model_convnext_temp_scale.json   # calibration T=0.73
└── archive/
    └── dog_model.h5               # old MobileNetV2 (optional)
```

> The calibration `.json` files and breed labels (`data/unique_breeds.json`) are already in the repo — you only need the two `.keras` files.

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

### 4. Open the app

- **Desktop:** `http://127.0.0.1:8000` — allow camera, tap the red button
- **Phone (same network):** `http://<YOUR_LAPTOP_IP>:8000` — the PWA works on mobile browsers
- **Single image test (curl):**

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@my_dog.jpg"
```

### 5. What to expect

- Tap the **red button** → camera opens (back camera on mobile, webcam on desktop)
- The detector takes **~3 seconds on the first frame** (TensorFlow graph compilation), then settles at **3–5 fps**
- Each detected dog gets a **colored bounding box** with its breed name and confidence percentage
- The **480p/360p/240p** button changes resolution (lower = faster, higher = more accurate detection)
- Use the **flip button** (&#x21bb;) to switch between front and back camera
- Boxes **persist for 2 seconds** with a smooth fade-out so you can actually read the label

---

## What You Get

| Feature | Description |
|---------|-------------|
| **Live camera detection** | WebSocket streams frames from your phone to the server in real time |
| **Dog bounding boxes** | Pre-trained EfficientDet-Lite0 (5MB) finds all dogs in each frame |
| **120 breed classification** | Each detected dog is classified by a 2-model ensemble (EfficientNetV2S + ConvNeXtTiny) |
| **Multi-dog support** | Multiple dogs detected and classified in a single frame — all crops processed in one batch |
| **PWA** (Progressive Web App) | Add to your phone's home screen — works like a native app with manifest, icon, and service worker |
| **Single image API** | `POST /predict` — upload an image, get breed prediction back |
| **Unknown detection** | Below 50% confidence → returns `"unknown"` instead of guessing |

---

## Project Structure

```
app/
├── main.py              # FastAPI app (routes, WebSocket, CORS, static files)
├── predictor.py         # DogBreedPredictor — ensemble, TTA, temperature scaling
├── preprocessing.py     # Image preprocessing (resize, normalize, TTA variants)
├── detector.py          # DogDetector — EfficientDet-Lite0 from TensorFlow Hub
├── pipeline.py          # LivePipeline — detection → crop → batch classification
└── static/
    ├── index.html       # PWA frontend (camera, canvas overlay, WebSocket client)
    ├── manifest.json    # PWA manifest (Add to Home Screen)
    ├── sw.js            # Service worker (network-first caching)
    └── icon.svg         # App icon (dog silhouette)
data/
└── unique_breeds.json   # 120 breed labels (snake_case)
models/
├── dog_model_improved.keras          # EfficientNetV2S (21M params)
├── dog_model_convnext.keras          # ConvNeXtTiny (28M params)
├── *_temp_scale.json                 # per-model temperature calibration
└── archive/                          # old MobileNetV2 (kept for reference)
Dockerfile               # Hugging Face Spaces / Docker deployment
deploy_hfspaces.sh       # One-command HF Spaces deploy script
requirements.txt         # 8 pinned dependencies
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | PWA frontend — camera + live detection UI |
| `GET` | `/api/health` | Health check — returns version and endpoints |
| `GET` | `/breeds` | List all 120 supported breeds |
| `POST` | `/predict` | Upload an image → breed prediction |
| `POST` | `/predict?tta=true` | Same with test-time augmentation (slower, slightly more accurate) |
| `WS` | `/ws/live` | WebSocket — send JPEG frames, receive `[{bbox, breed, confidence}]` per frame |

### Single Image Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@dog.jpg"
```

```json
{
  "primary": { "breed": "golden_retriever", "confidence": 0.95 },
  "top_k": [
    { "rank": 1, "breed": "golden_retriever", "confidence": 0.95 },
    { "rank": 2, "breed": "labrador_retriever", "confidence": 0.03 },
    { "rank": 3, "breed": "chesapeake_bay_retriever", "confidence": 0.01 }
  ],
  "is_unknown": false,
  "message": "Prediction successful.",
  "ensemble": {
    "num_models": 2,
    "all_agree": true,
    "individual_predictions": [
      { "breed": "golden_retriever", "confidence": 0.96 },
      { "breed": "golden_retriever", "confidence": 0.94 }
    ]
  }
}
```

### WebSocket Live Detection

**Client → Server:** Raw JPEG bytes (binary WebSocket frame, ~30KB at 480p)

**Server → Client:** JSON per frame

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
  "fps": 4.2,
  "width": 640,
  "height": 480
}
```

`bbox` is `[x1, y1, x2, y2]` in absolute pixel coordinates of the sent frame. `breed_conf` is breed classifier confidence; `det_conf` is detector confidence.

---

## Performance

| Metric | Local (M1/M2 Mac) | HF Spaces (8 vCPU) | HF Spaces (2 vCPU) |
|--------|:-:|:-:|:-:|
| First frame (TF warmup) | ~3s | ~5s | ~8s |
| Steady-state FPS (1 dog) | 4–5 | 4–5 | 1.5–2 |
| Steady-state FPS (2 dogs) | 3–4 | 3–4 | 1–2 |
| RAM usage | ~7GB | ~7GB | ~7GB |

The pipeline uses a **single-model fast path** (EfficientNetV2S, 21M params) for live mode instead of the full 2-model ensemble (49M params). This halves inference time with negligible accuracy loss for real-time use. All detected dog crops are classified in a single batched forward pass.

---

## Model Architecture

Each breed classifier shares the same head; only the backbone differs:

```
Input (224×224×3 RGB, values 0–1)
  → Rescaling(255)                 # backbones expect raw 0–255 pixels
  → Backbone (EfficientNetV2S or ConvNeXtTiny), ImageNet-pretrained
  → GlobalAveragePooling2D
  → Dropout(0.4)
  → Dense(512, ReLU)
  → Dropout(0.2)
  → Dense(120, softmax)
```

The `/predict` endpoint averages both models' temperature-calibrated outputs (ensemble). The live `/ws/live` pipeline uses the **smaller single model** (EfficientNetV2S) for speed.

---

## Deploying (Optional)

### Hugging Face Spaces

```bash
# From the project root:
./deploy_hfspaces.sh
```

This clones the HF Space repo, copies all files, sets up Git LFS for model weights, and pushes. Requires a HF PRO subscription ($9/mo) for Docker SDK support.

**Cost:** $0.03/hour (8 vCPU, 32GB) with 15-min auto-sleep. At ~1 hour/day = **~$1/month** + $9 PRO subscription.

### Docker

```bash
docker build -t dog-vision .
docker run -p 7860:7860 dog-vision
```

The Dockerfile pre-caches the EfficientDet-Lite0 detector from TF Hub during build and auto-downloads the breed models from the GitHub Release if not found locally.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` on import | Make sure you activated the venv: `source venv/bin/activate` |
| Models not found at startup | Download the `.keras` files from the [GitHub Release](https://github.com/MozzamShahid/dog-vision-back/releases/tag/v2.0-ensemble) into `models/` |
| Camera doesn't open | Must use `https://` or `localhost`. Mobile browsers require HTTPS for camera access |
| WebSocket 404 / "reconnecting" | Make sure `requirements.txt` has `uvicorn[standard]==0.49.0` (not bare `uvicorn`) |
| "No supported WebSocket library" | Install: `pip install 'uvicorn[standard]'` |
| First frame takes 10+ seconds | Normal — TensorFlow compiles the graph on first inference. Subsequent frames are fast |
| Detection feels slow | Lower resolution to 360p or 240p. The 2 vCPU free tier is slower — upgrade to 8 vCPU for 4+ fps |
| Memory error (OOM) | Models need ~7GB RAM. Free HF Spaces have 16GB, so should be fine |
| `libgl` errors on Linux | Install system deps: `apt install libgl1 libglib2.0-0 libgomp1` |

---

## The 120 Supported Breeds

affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel, bloodhound, bluetick, border_collie, border_terrier, borzoi, boston_bull, bouvier_des_flandres, boxer, brabancon_griffon, briard, brittany_spaniel, bull_mastiff, cairn, cardigan, chesapeake_bay_retriever, chihuahua, chow, clumber, cocker_spaniel, collie, curly-coated_retriever, dandie_dinmont, dhole, dingo, doberman, english_foxhound, english_setter, english_springer, entlebucher, eskimo_dog, flat-coated_retriever, french_bulldog, german_shepherd, german_short-haired_pointer, giant_schnauzer, golden_retriever, gordon_setter, great_dane, great_pyrenees, greater_swiss_mountain_dog, groenendael, ibizan_hound, irish_setter, irish_terrier, irish_water_spaniel, irish_wolfhound, italian_greyhound, japanese_spaniel, keeshond, kelpie, kerry_blue_terrier, komondor, kuvasz, labrador_retriever, lakeland_terrier, leonberg, lhasa, malamute, malinois, maltese_dog, mexican_hairless, miniature_pinscher, miniature_poodle, miniature_schnauzer, newfoundland, norfolk_terrier, norwegian_elkhound, norwich_terrier, old_english_sheepdog, otterhound, papillon, pekinese, pembroke, pomeranian, pug, redbone, rhodesian_ridgeback, rottweiler, saint_bernard, saluki, samoyed, schipperke, scotch_terrier, scottish_deerhound, sealyham_terrier, shetland_sheepdog, shih-tzu, siberian_husky, silky_terrier, soft-coated_wheaten_terrier, staffordshire_bullterrier, standard_poodle, standard_schnauzer, sussex_spaniel, tibetan_mastiff, tibetan_terrier, toy_poodle, toy_terrier, vizsla, walker_hound, weimaraner, welsh_springer_spaniel, west_highland_white_terrier, whippet, wire-haired_fox_terrier, yorkshire_terrier

---

Built by **Mozzam**.
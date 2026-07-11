# Dog Vision API

A FastAPI backend that classifies dog breeds from an uploaded image using an
ensemble of two TensorFlow models. Supports **120 breeds** (Stanford Dogs).

**Top-1 accuracy: 92.42%** (ensemble, on a held-out validation set)
**Top-3 accuracy: 99.30%** — the correct breed is in the top 3 almost every time.

📊 **[Model improvement report →](docs/model-improvements.html)** — the before/after story in one page.

---

## The short story (for anyone picking this up)

The classifier started as a single small model (MobileNetV2) with a training
pipeline that had silent bugs. It was rebuilt into a calibrated two-model
ensemble. On the same 1,280-image validation set:

| Model | Params | Top-1 | Top-3 | Top-5 |
|-------|-------:|------:|------:|------:|
| Old — MobileNetV2 | 2.4M | 88.05% | 96.80% | 97.89% |
| EfficientNetV2S | 21.0M | 89.92% | 98.67% | 99.45% |
| ConvNeXtTiny | 28.3M | 90.78% | 98.75% | 99.53% |
| **Ensemble (current)** | **49.3M** | **92.42%** | **99.30%** | **99.69%** |

**88.05% → 92.42% top-1** — about a third of the previous errors removed
(wrong answers fell from ~12% to ~7.6%).

### What changed

1. **Fixed a silent input-scaling bug.** Augmentation (brightness/contrast) was
   running *after* images were scaled to 0–1, turning training pictures into
   near-white noise. Every model trained on that data learned from garbage.
2. **Upgraded the backbone.** MobileNetV2 → EfficientNetV2S **and** ConvNeXtTiny,
   two stronger, more recent vision models.
3. **Ensembled the two models.** The API averages both predictions; where one
   hesitates the other often decides, so the ensemble beats either alone.
4. **Calibrated confidence.** Temperature scaling (T=0.67 / T=0.73) makes each
   model's reported confidence match how often it is actually right.
5. **Made evaluation trustworthy.** The old figure came from 25 hand-picked
   images; every number here is measured on a 4,116-image held-out set.

---

## Environment

- **Python** 3.11
- **FastAPI** 0.139.0
- **TensorFlow** 2.21.0
- **TensorFlow Hub** 0.16.1

## Project structure

```
app/
├── main.py            # FastAPI app (GET /, POST /predict, GET /breeds)
├── predictor.py       # DogBreedPredictor — ensemble, TTA, temperature scaling
└── preprocessing.py   # Image preprocessing (224x224, TTA variants)
data/
└── unique_breeds.json # 120 breed labels
models/
├── dog_model_improved.keras          # EfficientNetV2S
├── dog_model_convnext.keras          # ConvNeXtTiny
├── *_temp_scale.json                 # per-model confidence calibration
└── archive/dog_model.h5              # old MobileNetV2 (kept for reference)
docs/
└── model-improvements.html           # visual before/after report
train.py                # local training / fine-tuning script
kaggle_train.py         # GPU training script (run on Kaggle)
reeval_cell.py          # re-evaluate saved models + export clean .keras files
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the server

```bash
uvicorn app.main:app --reload
```

Available at `http://127.0.0.1:8000`. On startup the predictor auto-discovers
every `.keras` (and legacy `.h5`) model in `models/` and ensembles them.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/breeds` | List all 120 supported breeds |
| POST | `/predict` | Upload an image, returns breed prediction |
| POST | `/predict?tta=true` | Prediction with test-time augmentation (slower, a bit more accurate) |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@dog.jpg"
```

```json
{
    "primary": { "breed": "golden_retriever", "confidence": 0.95 },
    "top_k": [
        {"rank": 1, "breed": "golden_retriever", "confidence": 0.95},
        {"rank": 2, "breed": "labrador_retriever", "confidence": 0.03},
        {"rank": 3, "breed": "chesapeake_bay_retriever", "confidence": 0.01}
    ],
    "is_unknown": false,
    "message": "Prediction successful.",
    "ensemble": {
        "num_models": 2,
        "all_agree": true,
        "individual_predictions": [
            {"breed": "golden_retriever", "confidence": 0.96},
            {"breed": "golden_retriever", "confidence": 0.94}
        ]
    }
}
```

When top confidence is below 50%, `primary.breed` becomes `"unknown"` and
`is_unknown` is `true` — the breed may not be one of the 120 we support.

## Model architecture

Each model shares the same head; only the backbone differs:

```
Input (224×224×3, values 0–1)
  → Rescaling(255)          # backbones expect raw 0–255; baked into the graph
  → Backbone (EfficientNetV2S  or  ConvNeXtTiny), ImageNet-pretrained
  → GlobalAveragePooling2D
  → Dropout(0.4)
  → Dense(512, relu)
  → Dropout(0.2)
  → Dense(120, softmax)
```

The API averages the two models' softmax outputs (each temperature-calibrated
first) to produce the final prediction.

## Training

Training runs on a GPU (Kaggle). The pipeline:

- **Data augmentation** on raw pixels — flips, rotation, zoom, brightness,
  contrast, translation
- **Two-step schedule** — train the classifier head with the backbone frozen,
  then unfreeze the last ~30% and fine-tune at a low learning rate
- **MixUp / CutMix** during fine-tuning for regularization
- **Label smoothing** to reduce overconfidence on similar breeds
- **Temperature scaling** after training for calibrated confidence

```bash
# Local (CPU/GPU)
python train.py --data_dir /path/to/stanford-dogs --output models/dog_model_improved.keras

# Kaggle GPU — see kaggle_train.py (trains both backbones end to end)
```

> **Note on model files.** `.keras` is the format to use. ConvNeXt does not
> reload cleanly from legacy `.h5` under Keras 3, so `predictor.py` rebuilds the
> architecture and loads weights as a fallback — but the `.keras` files load
> directly. `reeval_cell.py` regenerates the clean `.keras` files from Kaggle
> `.h5` checkpoints.

## Future ideas

- [ ] Progressive resizing to 384px (~+1% for ~2× training time)
- [ ] Expand beyond 120 breeds
- [ ] Breed description / metadata endpoint

---

Built and improved by **Mozzam**.

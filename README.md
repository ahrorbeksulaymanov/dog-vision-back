# Dog Vision API

A FastAPI backend that uses TensorFlow to classify dog breeds from uploaded images.

**Current accuracy:** 89.5% on known breeds (see [TESTING.md](TESTING.md))
**Supported breeds:** 120

## Environment

- **Python**: 3.11.14 (via `venv/`)
- **FastAPI**: 0.139.0
- **TensorFlow**: 2.21.0
- **TensorFlow Hub**: 0.16.1

## Project Structure

```
app/
├── main.py            # FastAPI app (GET /, POST /predict, GET /breeds)
├── predictor.py       # DogBreedPredictor (top-3, TTA, confidence threshold)
└── preprocessing.py   # Image preprocessing (224x224, TTA variants)
data/
└── unique_breeds.json # 120 breed labels
models/
└── dog_model.h5       # Trained Keras model
train.py               # Training & fine-tuning script
TESTING.md             # Model testing report
```

## Setup

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/breeds` | List all 120 supported breeds |
| POST | `/predict` | Upload an image, returns breed prediction |
| POST | `/predict?tta=true` | Prediction with test-time augmentation (slower, more accurate) |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@dog.jpg"
```

Response:

```json
{
    "primary": {
        "breed": "golden_retriever",
        "confidence": 0.95
    },
    "top_k": [
        {"rank": 1, "breed": "golden_retriever", "confidence": 0.95},
        {"rank": 2, "breed": "labrador_retriever", "confidence": 0.03},
        {"rank": 3, "breed": "chesapeake_bay_retriever", "confidence": 0.01}
    ],
    "is_unknown": false,
    "message": "Prediction successful."
}
```

When confidence is below 50%:

```json
{
    "primary": {
        "breed": "unknown",
        "confidence": 0.18
    },
    "top_k": [...],
    "is_unknown": true,
    "message": "Confidence too low — breed may not be in our database."
}
```

## Training / Fine-tuning

To improve the model, download the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and run:

```bash
python train.py --data_dir /path/to/dataset
```

The training script includes:
- **Data augmentation** — random flips, rotations, zoom, brightness, contrast
- **Two-phase training** — classifier head first, then unfreeze backbone
- **Label smoothing** — reduces overconfidence on similar breeds
- **Learning rate scheduling** — ReduceLROnPlateau
- **Early stopping** — prevents overfitting
- **Top-3 accuracy tracking** — monitors top-3 accuracy during training

Options:

```
--data_dir DIR       Path to dataset (organized by breed subfolders)
--epochs N           Number of training epochs (default: 30)
--batch_size N       Batch size (default: 32)
--unfreeze_at N      Epoch to unfreeze backbone (default: 10)
--output PATH        Output model path (default: models/dog_model_improved.h5)
--resume PATH        Resume from existing model
```

## Model Architecture

```
MobileNetV2 (pretrained on ImageNet)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(120, softmax)
```

Total params: 2.4M | Trainable: 154K (head only) → 1.2M (after unfreeze)

## Future Improvements

- [ ] Upgrade backbone to EfficientNetV2
- [ ] Expand to 200+ breeds
- [ ] Add breed description/metadata endpoint
- [ ] Multi-model ensemble
- [ ] Confidence calibration (temperature scaling)
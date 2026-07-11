# Dog Vision — Architecture & Design

> Comprehensive reference for the Dog Vision backend: what it is, the
> technologies it uses, how it's put together, what each model-improvement
> technique contributes, and what is *written* vs *trained* vs *deployed*.
>
> Last updated: 2026-07-09

---

## 1. Project Overview

**Dog Vision** is a **dog breed classification API**. A client uploads an image
of a dog; the backend returns the predicted breed from **120 breeds** of the
Stanford Dogs Dataset, along with top-3 predictions and an "unknown" flag for
low-confidence / out-of-distribution inputs.

| Metric | Value |
|---|---|
| Breeds supported | 120 (Stanford Dogs Dataset) |
| Current accuracy | **89.5%** (17/19 known-breed tests, per `TESTING.md`) |
| Model file committed | `models/dog_model.h5` (~11 MB, MobileNetV2-based) |
| API version | 2.0.0 (`app/main.py`) |

---

## 2. Tech Stack

| Layer | Technology | Version (pinned) | Purpose |
|---|---|---|---|
| Language | Python | 3.11.11 (`runtime.txt`, `.python-version`) | Runtime |
| Web framework | FastAPI | 0.139.0 | REST API |
| ASGI server | Uvicorn | 0.49.0 | Serves FastAPI |
| Multipart parsing | python-multipart | 0.20 | `UploadFile` form support |
| Deep learning | **TensorFlow / Keras** | 2.21.0 | Model load, training, inference |
| Transfer-learning source | TensorFlow Hub | 0.16.1 | Hub `KerasLayer` `custom_objects` on `.h5` load |
| Image I/O | Pillow | 11.3.0 | Decode / resize / crop / flip |
| Numeric ops | NumPy | 2.4.6 | Batch stacking, ensemble mean, top-k argsort |

**Notable absences:** no PyTorch, no Dockerfile, no `Procfile`, no tests /
pytest, no CI, no CORS, no auth / rate-limiting. The only deployment hint is
the Heroku-style `runtime.txt`.

---

## 3. High-Level Architecture

```
                    ┌──────────────────────────────────────────┐
   Client ──POST /predict (image, ?tta=true)──► FastAPI app/main.py
                    │                                          │
                    │   GET /              health check         │
                    │   GET /breeds        120 breed list        │
                    └──────────────┬───────────────────────────┘
                                   │
                                   ▼
                       DogBreedPredictor (singleton, loaded at import)
                       app/predictor.py
                                   │
                  ┌────────────────┼─────────────────┐
                  ▼                ▼                  ▼
           preprocess_image   preprocess_image_tta   np.mean over
           (224×224, /255)    (10 variants)         ensemble models
                  │                │                  │
                  └────────────────┴──────────────────┘
                                   │
                                   ▼
                  One or more .h5 (auto-loaded from /models)
                  MobileNetV2 backbone + custom head
                                   │
                                   ▼
                  Top-3 + confidence + is_unknown (< 0.50)
                                   │
                                   ▼
                  JSON {primary, top_k, is_unknown, ensemble?}
```

**Offline training side (`train.py`):**

```
   Stanford Dogs Dataset (data/Images/<breed>/*.jpg)
            │
            ▼
   image_dataset_from_directory (val_split=0.2, seed=42, categorical)
            │
            ▼
   Augmentation: flip / rot / zoom / brightness / contrast / translation
     + 50/50 MixUp(α=0.2) / CutMix(α=0.2)
            │
            ▼
   Backbone {MobileNetV2 | EfficientNetV2B0 | V2S | ConvNeXtTiny}
     (ImageNet weights; frozen → partial unfreeze)
            │
            ▼
   GAP → Dropout(0.4) → Dense(512) → Dropout(0.2) → Dense(120, softmax)
            │
            ▼
   Loss: CategoricalCrossentropy(label_smoothing 0.1 → 0.05)
   Opt:  AdamW (lr 1e-3 head, 5e-6..1e-5 fine-tune), weight_decay 1e-4
   Metrics: accuracy, top-3, top-5
            │
            ▼
   Callbacks: EarlyStopping(p=8) + ReduceLROnPlateau(p=4) + ModelCheckpoint(best)
            │
            ▼
   Phase 2 (optional --progressive): 224 → 384, transfer weights, fine-tune
            │
            ▼
   models/dog_model_improved.h5
```

---

## 4. Application Structure

```
dog-vision-back/
├── runtime.txt              heroku-style python-3.11.11
├── .python-version          3.11.11
├── requirements.txt         7 pinned deps
├── README.md                project docs (now stale vs train.py)
├── TESTING.md               manual evaluation report
├── ARCHITECTURE.md          this file
├── train.py                 full training pipeline (486 lines)
├── data/
│   └── unique_breeds.json   120 breed labels (snake_case)
├── models/
│   └── dog_model.h5         trained Keras model (~11 MB, MobileNetV2)
└── app/                     FastAPI application
    ├── main.py             FastAPI app + 3 routes
    ├── predictor.py        DogBreedPredictor + ensemble + TTA
    └── preprocessing.py    preprocess_image, preprocess_image_tta
```

| File | Role | Key symbols |
|---|---|---|
| `app/main.py:8` | FastAPI app, routes | `app`, `home`, `list_breds`, `predict` |
| `app/predictor.py` | Inference + ensemble + unknown detection | `DogBreedPredictor`, `_find_models`, `predictor` (singleton) |
| `app/preprocessing.py` | Image prep + TTA variants | `preprocess_image`, `preprocess_image_tta`, `_center_crop` |
| `train.py` | Training pipeline | `parse_args`, `BACKBONES`, `build_model`, `mixup_batch`, `cutmix_batch`, `build_data_pipeline`, `train` |
| `data/unique_breeds.json` | 120 breed labels | snake_case Stanford Dogs names |
| `models/dog_model.h5` | Trained Keras model | MobileNetV2 backbone + head |

**Why the model loads at boot:** `app/predictor.py` instantiates the singleton
`predictor = DogBreedPredictor(...)` at module-import time. Servers pay the
TensorFlow + `.h5` load cost on cold start, after which inference is in-memory.

---

## 5. API Endpoints

| Method | Path | Signature | Returns |
|---|---|---|---|
| GET | `/` | — | `{message, version, endpoints}` health check |
| GET | `/breeds` | — | `{total: 120, breeds: [...]}` |
| POST | `/predict` | `file: UploadFile, tta: bool = False` | `{primary, top_k, is_unknown, message, ensemble?}` |

`POST /predict` JSON response shape:

```json
{
  "primary": { "rank": 1, "breed": "golden_retriever", "confidence": 0.93 },
  "top_k": [
    { "rank": 1, "breed": "golden_retriever", "confidence": 0.93 },
    { "rank": 2, "breed": "...", "confidence": 0.04 },
    { "rank": 3, "breed": "...", "confidence": 0.01 }
  ],
  "is_unknown": false,
  "message": "Predicted breed: golden_retriever",
  "ensemble": {                  // present only when >1 model loaded
    "all_agree": true,
    "per_model": [...]
  }
}
```

---

## 6. Model Architecture

### Backbones (`train.py` `BACKBONES`)

| `--backbone` | Keras application | `unfreeze_at` | Fine-tune LR | Params |
|---|---|---|---|---|
| `mobilenetv2` (default) | MobileNetV2 | 100 | 1e-5 | small |
| `efficientnetv2` | EfficientNetV2B0 | 150 | 5e-6 | small-medium |
| `efficientnetv2s` | EfficientNetV2S | 200 | 5e-6 | medium |
| `convnext` | ConvNeXtTiny | 180 | 5e-6 | medium |

All use `IMG_SIZE = 224`, ImageNet weights, `include_top=False`.

### Classifier head (`build_model`)

```
Input(224×224×3)
  → backbone(trainable=False initially)
  → GlobalAveragePooling2D
  → Dropout(0.4)
  → Dense(512, relu)
  → Dropout(0.2)
  → Dense(num_classes=120, softmax)
```

> ⚠️ **README drift:** `README.md` documents the simpler original architecture
> (`MobileNetV2 → GAP → Dropout(0.3) → Dense(120)`). The current `train.py`
> uses the richer head above and supports 4 backbones. The committed
> `models/dog_model.h5` corresponds to the **older, simpler** architecture in
> the README, not the current `train.py`.

---

## 7. Training Pipeline (`train.py`)

### Dataset
- **Stanford Dogs Dataset** — `data_dir/<breed_name>/*.jpg`, one folder per breed.
- Loaded via `tf.keras.utils.image_dataset_from_directory` with
  `validation_split=0.2`, `seed=42`, `label_mode="categorical"`, `shuffle=True`.
- Class names auto-derived from folder names and **saved to
  `data/unique_breeds.json`**.
- The full image dataset is **not committed**; `train.py` validates the
  directory and prints Stanford Dogs download instructions if missing.

### CLI arguments (`parse_args`)
| Arg | Default | Notes |
|---|---|---|
| `--data_dir` | `data/Images` | Stanford Dogs layout |
| `--epochs` | 30 | |
| `--batch_size` | 32 | |
| `--backbone` | `mobilenetv2` | one of the 4 above |
| `--progressive` | off | enables 224→384 two-phase |
| `--mixup` | on (`--no-mixup` to disable) | MixUp + CutMix |
| `--output` | `models/dog_model_improved.h5` | |
| `--resume` | — | continue from an existing `.h5` |

### Data augmentation (training only)
Keras `Sequential` layer applied to the training set:
`RandomFlip("horizontal")`, `RandomRotation(0.15)`, `RandomZoom(0.15)`,
`RandomBrightness(0.15)`, `RandomContrast(0.15)`, `RandomTranslation(0.1, 0.1)`,
then `Rescaling(1/255)`, `prefetch(AUTOTUNE)`.

### MixUp / CutMix
- `mixup_batch(images, labels, alpha=0.2)` — symmetric λ blend of image+label pairs.
- `cutmix_batch(images, labels, alpha=0.2)` — vectorized patch cut-paste with λ recomputed from the actual cut area.
- Applied per-batch with 50/50 random selection (when `--mixup` is on, which is the default).

### Training procedure (`train`)
- **Phase 1** (224×224):
  - **Step A — Train head** (backbone frozen) for `head_epochs = max(5, epochs // 4)`.
  - **Step B — Fine-tune** partial-unfreeze backbone above `unfreeze_at`, remaining epochs, **lower LR**, label smoothing reduced from 0.1 → 0.05.
- **Phase 2 (optional, `--progressive`):**
  - Rebuild data pipeline at **384×384** with `batch_size // 2`.
  - Rebuild model at 384 input; transfer weights layer-by-layer from the 224 model.
  - Fine-tune for `epochs // 3` more, LR halved.
- Final evaluation prints val_accuracy, top-3, top-5, loss; saves with `model.save(args.output)`.

### Optimizer / loss / metrics
- **Optimizer:** `AdamW` — lr `1e-3` for head training, `BACKBONES[...]["lr"]` (5e-6..1e-5) for fine-tune; `weight_decay=1e-4`.
- **Loss:** `CategoricalCrossentropy(label_smoothing=0.1)` in Step A, `0.05` in Step B.
- **Metrics:** `accuracy`, `TopKCategoricalAccuracy(k=3, name="top3_acc")`, `TopKCategoricalAccuracy(k=5, name="top5_acc")`.

### Callbacks
- `EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7)`
- `CSVLogger("training_log.csv")`
- `ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True)` (Step B)

---

## 8. Inference Pipeline (`app/`)

### Preprocessing (`app/preprocessing.py`)
- **`preprocess_image(image_bytes)`** — PIL decode → RGB → resize 224×224 → float32 → `/255` → `(1,224,224,3)`.
- **`preprocess_image_tta(image_bytes, num_augments=8)`** — generates up to **10 variants** (original center crop, horizontal flip, 4 corner crops + their flips), each resized 256 then center-cropped to 224, stacked to `(N,224,224,3)`, normalized.

### Prediction (`app/predictor.py`)
- `DogBreedPredictor.__init__(model_paths, labels_path, confidence_threshold=0.50, top_k=3)` accepts a **single path or a list** (ensemble). Each `.h5` is loaded with `custom_objects={"KerasLayer": hub.KerasLayer}`.
- `predict(image_bytes, use_tta=False)`:
  - With TTA, averages predictions across the 10 variants (`np.mean(preds, axis=0)`).
  - **Ensemble:** if multiple models, `np.mean(all_predictions, axis=0)`; computes `all_agree` (do all models agree on the top breed?) and returns per-model predictions.
  - Top-K via `np.argsort` → `top_k` list of `{rank, breed, confidence}`.
  - `is_unknown = primary.confidence < 0.50` → primary breed becomes `"unknown"`.
- `_find_models(models_dir="models")` auto-loads **every `.h5`** in `models/` — adding trained models there silently activates ensemble mode.
- **Singleton** `predictor = DogBreedPredictor(model_paths=_find_models(), confidence_threshold=0.50, top_k=3)` instantiated at module import → cold-start cost.

---

## 9. Feature Impact Table

Each technique used, the expected accuracy/quality impact, and a **Status**
flag so it's clear what is merely *written* vs actually *trained* vs *deployed*.

**Status legend:**

- 🟢 **Deployed** — runs today from the committed `models/dog_model.h5`.
- 🟡 **Trained only** — implemented in code; needs an improved `.h5` to take effect.
- ⚪ **Written** — code present in `train.py`/`app/`, but no trained artifact exists yet.

| # | Feature | Where | Expected acc impact | Other impact | Status |
|---|---|---|---|---|---|
| 1 | Transfer learning (MobileNetV2 + ImageNet) | `train.py` backbone | baseline (~80–85%) | Tiny model (~11 MB), fast | 🟢 |
| 2 | Top-3 return (not just top-1) | `predictor.py` | UX; top-3 ~95% even when top-1 ~85% | Better ranking visibility | 🟢 |
| 3 | Confidence threshold 0.50 → "unknown" | `predictor.py` | Reduces false positives on out-of-distribution images | Robustness for real-world use | 🟢 |
| 4 | Test-time augmentation (10 variants averaged) | `preprocessing.py`, `predictor.py` | **+1–3%** | ~10× inference latency when enabled | 🟢 (code) / ⚪ (no improved .h5 to TTA) |
| 5 | Multi-model ensemble (auto-load `.h5`s) | `predictor.py` `_find_models` | **+0.5–2%**, esp. on edge cases | Linear cost; gives `all_agree` flag | ⚪ only one `.h5` in `models/` today |
| 6 | Backbone partial-unfreeze fine-tuning | `train.py` Step B | **+2–5%** over frozen head | Needs low LR + EarlyStopping | 🟡 trained — but not saved into a new `.h5` |
| 7 | Rich augmentation (flip/rot/zoom/brightness/contrast/translation) | `train.py` aug | **+2–4%** | Better generalization, less overfit | 🟡 |
| 8 | MixUp + CutMix (α=0.2, 50/50) | `train.py` | **+1–3%**, esp. on small per-class data | Harder labels; pairs with label smoothing | 🟡 |
| 9 | Label smoothing (0.1 → 0.05) | `train.py` loss | **+0.5–1%**, better calibration | Reduces overconfidence | 🟡 |
| 10 | AdamW + ReduceLROnPlateau + EarlyStopping | `train.py` callbacks | indirect **+1–2%** via stability | Faster convergence, auto lr decay | 🟡 |
| 11 | Multi-backbone support (EffV2B0 / V2S / ConvNeXtTiny) | `train.py` `BACKBONES` | EffV2S / ConvNeXt can give **+2–4%** | Large models; more memory/time | ⚪ none trained yet |
| 12 | Progressive resizing (224 → 384) | `train.py` Phase 2 `--progressive` | **+1–2%** in final stage | Slower final phase; avoids loss spikes | ⚪ not run yet |
| 13 | Top-3 / Top-5 metrics | `train.py` compilation | none on accuracy | Better visibility into ranking | 🟢 (runs when training) |

**Rough cumulative ceiling** if every technique fires ideally: from a vanilla
~80% MobileNetV2 baseline up to ~90–93% top-1 — consistent with the current 89.5%.

### What this means in practice today
- The committed `models/dog_model.h5` only reflects the **older** simpler
  architecture (#1, #2, #3, and a MobileNetV2 transfer head).
- All of `train.py`'s advanced features (#6–#13) are **written** but have
  produced **no committed improved model** yet. Running
  `python train.py --backbone efficientnetv2s --progressive` is the single
  highest-impact next step.
- Once an improved `.h5` lands in `models/`, features #4–#5 activate
  **automatically** with no code change (`_find_models` picks them up).

---

## 10. Current Status & Gaps

### What's committed (in git, on `main`)
- Original `app/main.py`, `app/predictor.py`, `app/preprocessing.py`
  (simpler versions).
- `models/dog_model.h5` (original MobileNetV2 model).
- `data/unique_breeds.json`, `requirements.txt`, `runtime.txt`.

### What's uncommitted locally (done but not pushed)
- Expanded `app/main.py` (+36 lines: `/breeds`, versioning).
- Greatly expanded `app/predictor.py` (+158 lines: ensemble, TTA integration,
  auto-detect models, `is_unknown`).
- Greatly expanded `app/preprocessing.py` (+54 lines: TTA variants).
- **Untracked:** `train.py` (entire file), `README.md`, `TESTING.md`.

> So everything described in §6–§9 exists in the working tree but is **not yet
> in the repo history**.

### Known gaps
1. **No improved trained model committed.** `models/` contains only the original
   `dog_model.h5`. The ensemble / TTA-gains from `predictor.py` have no improved
   `.h5` to operate on yet.
2. **README ↔ code drift.** README documents the old MobileNetV2 + Dropout(0.3) + Dense(120) head; `train.py` uses the richer head and 4 backbones.
3. **No deployment config.** No `Procfile` / `Dockerfile` / `render.yaml`. Heroku detection would be fragile; the ~11 MB committed model and the TensorFlow runtime will push slug/image size. Render Web Service or Docker (Fly.io / Railway) are better fits.
4. **No CORS middleware.** A frontend at a different origin cannot call this API.
5. **No automated tests / no CI.** `TESTING.md` is manual. No `pytest`, no lint config, no GitHub Actions.
6. **Cold-start cost.** The `predictor` singleton loads the model at import time; first request after boot pays the full TF load time.
7. **No auth / rate-limiting.** Suitable as an internal backend; would need hardening for public deployment.
8. **No experiment tracking.** Only `training_log.csv` is produced; no MLflow / W&B for run comparison.
9. **No calibration (temperature scaling).** Listed as a future improvement in README; not in `train.py`. Would make the 0.50 unknown-threshold more reliable.

---

## 11. Growth Roadmap (prioritized by impact / effort)

| Priority | Action | Est. impact | Unlocks / addresses |
|---|---|---|---|
| **P0** | **Train an improved model**: `python train.py --backbone efficientnetv2s --progressive --epochs 30` and place the resulting `.h5` in `models/`. | +2–4% accuracy | Activates ensemble+TTA code paths already in `predictor.py` |
| **P0** | **Commit uncommitted work** (`app/*` expansions + `train.py`, `README.md`, `TESTING.md`). | reproducibility | Makes the repo reflect reality |
| **P1** | **Train a 2nd backbone** (e.g. `convnext`) → true ensemble (`_find_models` auto-loads it). | +0.5–2% + robustness | Feature #5 |
| **P1** | **Add deploy config** — `Procfile` (`web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`) and optionally a `Dockerfile`; pick Render/Fly. | production-ready | Gap #3 |
| **P1** | **Add `CORSMiddleware`** for the named frontend origin. | enables frontend | Gap #4 |
| **P2** | **Temperature scaling** on a held-out set → calibrate confidences. | better unknown threshold | Gap #9 |
| **P2** | **pytest smoke tests** for `GET /`, `GET /breeds`, and a sample `POST /predict` against the committed model. | prevents regressions | Gap #5 |
| **P2** | **MLflow or W&B tracking** in `train.py`. | run comparability | Gap #8 |
| **P2** | **Sync README** with the real `train.py` architecture. | doc accuracy | Gap #2 |
| **P3** | Expand to **200+ breeds** (extended dataset). | coverage | |
| **P3** | Cache preprocessed image embeddings for faster TTA/ensemble. | inference speed | Gap #6 |

---

### Quick references (file:line)
- API entry: `app/main.py:8` (`app = FastAPI(...)`), `app/main.py` routes `home`, `list_breds`, `predict`.
- Predictor: `app/predictor.py` class `DogBreedPredictor`; `_find_models`; singleton `predictor` at module base.
- Preprocessing: `app/preprocessing.py` `preprocess_image`, `preprocess_image_tta`, `_center_crop`.
- Training: `train.py` `parse_args`, `BACKBONES`, `build_model`, `mixup_batch`, `cutmix_batch`, `build_data_pipeline`, `train`.
- Labels: `data/unique_breeds.json` (120 snake_case breed names).
- Model: `models/dog_model.h5` (~11 MB, committed).
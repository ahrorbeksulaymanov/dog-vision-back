# Model Testing Report

**Date:** 2026-07-09
**Test description:** 25 diverse dog images tested against the current model (MobileNetV2 frozen → Dense(120)).

## Test Setup

- **Images sourced from:** [Dog CEO API](https://dog.ceo/dog-api/) (public domain dog images)
- **Total images tested:** 25
- **Known breed images:** 19 (breed matches one of the 120 model labels)
- **Unknown breed images:** 6 (breed not in model labels, or random dog)

## Results Summary

| Metric | Value |
|---|---|
| **Accuracy (known breeds)** | 17/19 — **89.5%** |
| **Avg confidence** | 88.3% |
| **Median confidence** | 93.4% |
| **Low confidence (<60%)** | 2/25 (8%) |
| **High confidence (>90%)** | 13/25 (52%) |

## Detailed Results

### Correctly Classified (17/19)

| Image | Expected | Predicted | Confidence |
|---|---|---|---|
| beagle.jpg | Beagle | Beagle | 99.9% |
| boxer.jpg | Boxer | Boxer | 89.5% |
| chihuahua.jpg | Chihuahua | Chihuahua | 98.8% |
| chow.jpg | Chow | Chow | 100.0% |
| collie_border.jpg | Border Collie | Border Collie | 100.0% |
| doberman.jpg | Doberman | Doberman | 85.8% |
| keeshond.jpg | Keeshond | Keeshond | 98.2% |
| labrador.jpg | Labrador Retriever | Labrador Retriever | 78.3% |
| lhasa.jpg | Lhasa | Lhasa | 93.4% |
| malamute.jpg | Malamute | Malamute | 53.5% |
| newfoundland.jpg | Newfoundland | Newfoundland | 82.0% |
| pekinese.jpg | Pekinese | Pekinese | 96.3% |
| pomeranian.jpg | Pomeranian | Pomeranian | 99.8% |
| rottweiler.jpg | Rottweiler | Rottweiler | 99.4% |
| samoyed.jpg | Samoyed | Samoyed | 99.9% |
| vizsla.jpg | Vizsla | Vizsla | 99.6% |
| whippet.jpg | Whippet | Whippet | 87.2% |

### Misclassified (2/19)

| Image | Expected | Predicted | Confidence | Top-3 |
|---|---|---|---|---|
| papillon.jpg | Papillon | **Shetland Sheepdog** | 98.3% | shetland_sheepdog (98.3%), collie (1.0%), border_collie (0.2%) |
| pug.jpg | Pug | **French Bulldog** | 85.2% | french_bulldog (85.2%), pug (14.2%), schipperke (0.3%) |

### Unknown/Unlabeled Breeds (6)

| Image | Predicted | Confidence | Top-3 |
|---|---|---|---|
| dalmatian.jpg | Basset | 69.0% | basset (69.0%), boxer (13.8%), english_setter (6.1%) |
| random_1.jpg | Chow | 100.0% | chow (100.0%), tibetan_mastiff (0.0%), newfoundland (0.0%) |
| random_2.jpg | Norfolk Terrier | 69.6% | norfolk_terrier (69.6%), cairn (10.1%), border_terrier (9.6%) |
| random_3.jpg | Chesapeake Bay Retriever | 88.9% | chesapeake_bay_retriever (88.9%), bloodhound (3.7%), labrador_retriever (2.9%) |
| random_4.jpg | Norwegian Elkhound | 86.8% | norwegian_elkhound (86.8%), siberian_husky (5.4%), eskimo_dog (4.6%) |
| random_5.jpg | Miniature Pinscher | 48.5% | miniature_pinscher (48.5%), australian_terrier (22.5%), schipperke (9.1%) |

## Root Cause Analysis

### 1. Overconfident Misclassifications
The model predicted **98.3% confidence** for Papillon → Shetland Sheepdog. Both are small, fluffy dogs with pointed ears. MobileNetV2 (frozen on ImageNet features) cannot distinguish fine-grained details between similar breeds.

**Fix:** Fine-tune the backbone layers, add label smoothing during training.

### 2. Breed Not in Model
Dalmatian is not one of the 120 breeds. The model confidently returns "Basset" instead of indicating uncertainty.

**Fix:** Return "Unknown breed" when confidence is below a threshold. Expand breed coverage.

### 3. Low Confidence on Some Images
random_5.jpg got only 48.5% confidence — the model was uncertain, which is actually correct behavior for a difficult image.

**Fix:** Surface this uncertainty to the user via top-3 predictions.

### 4. 100% Confidence on Random Images
random_1.jpg got 100% confidence for "Chow" — this is a calibration issue. The softmax output is poorly calibrated.

**Fix:** Temperature scaling, confidence calibration.

## Improvement Plan

| # | Improvement | Type | Impact | Status |
|---|---|---|---|---|
| 1 | Top-3 predictions | Inference | Better UX | Done |
| 2 | Confidence threshold | Inference | Catch unknown breeds | Done |
| 3 | Test-time augmentation (TTA) | Inference | +2-3% accuracy | Done |
| 4 | Fine-tune backbone | Training | +5-10% accuracy | Done |
| 5 | Data augmentation | Training | Better generalization | Done |
| 6 | Learning rate scheduling | Training | Better convergence | Done |
| 7 | Label smoothing | Training | Reduce overconfidence | Done |
| 8 | Upgrade backbone (EfficientNetV2, ConvNeXt) | Architecture | +5-8% accuracy | Done |
| 9 | Model ensemble | Inference | +2-3% accuracy | Done |
| 10 | MixUp/CutMix | Training | +1-3% accuracy | Done |
| 11 | Progressive resizing | Training | +1-2% accuracy | Done |
| 12 | Expand breed coverage | Data | Cover more breeds | Future |
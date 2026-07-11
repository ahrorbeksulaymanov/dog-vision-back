"""
Dog Vision — Reusable Evaluation Script
=======================================

Tests the /predict endpoint with a set of dog images, with and without TTA,
and reports accuracy, confidence distribution, and per-image results.

Usage:
    # Start the server first:
    uvicorn app.main:app --reload

    # Then run this script:
    python tests/eval.py

    # With a custom API URL:
    python tests/eval.py --api_url http://localhost:8000

    # Without TTA (baseline only):
    python tests/eval.py --no-tta

The script caches test images locally so re-runs are identical.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.request

import requests

API_URL = "http://127.0.0.1:8000"
TEST_DIR = os.path.join(os.path.dirname(__file__), "images")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# The 25 test images from TESTING.md — breed name maps to the API label
# Known breeds are in the 120-class label set; unknown breeds test
# the confidence threshold.
TEST_IMAGES = [
    # Known breeds (19) — expected to be classified correctly
    {"file": "beagle.jpg",           "expected": "beagle",                "known": True},
    {"file": "boxer.jpg",            "expected": "boxer",                  "known": True},
    {"file": "chihuahua.jpg",        "expected": "chihuahua",              "known": True},
    {"file": "chow.jpg",             "expected": "chow",                   "known": True},
    {"file": "collie_border.jpg",    "expected": "border_collie",         "known": True},
    {"file": "doberman.jpg",         "expected": "doberman",               "known": True},
    {"file": "keeshond.jpg",         "expected": "keeshond",               "known": True},
    {"file": "labrador.jpg",         "expected": "labrador_retriever",     "known": True},
    {"file": "lhasa.jpg",            "expected": "lhasa",                  "known": True},
    {"file": "malamute.jpg",         "expected": "malamute",              "known": True},
    {"file": "newfoundland.jpg",     "expected": "newfoundland",          "known": True},
    {"file": "pekinese.jpg",         "expected": "pekinese",              "known": True},
    {"file": "pomeranian.jpg",       "expected": "pomeranian",            "known": True},
    {"file": "rottweiler.jpg",       "expected": "rottweiler",            "known": True},
    {"file": "samoyed.jpg",          "expected": "samoyed",               "known": True},
    {"file": "vizsla.jpg",           "expected": "vizsla",                "known": True},
    {"file": "whippet.jpg",          "expected": "whippet",               "known": True},
    # Previously misclassified — track specifically
    {"file": "papillon.jpg",         "expected": "papillon",              "known": True},
    {"file": "pug.jpg",              "expected": "pug",                    "known": True},
    # Unknown breeds (6) — should trigger is_unknown or different breed
    {"file": "dalmatian.jpg",        "expected": None,                    "known": False},
    {"file": "random_1.jpg",         "expected": None,                    "known": False},
    {"file": "random_2.jpg",         "expected": None,                    "known": False},
    {"file": "random_3.jpg",         "expected": None,                    "known": False},
    {"file": "random_4.jpg",         "expected": None,                    "known": False},
    {"file": "random_5.jpg",         "expected": None,                    "known": False},
]

# Dog CEO API breed → filename mapping for downloading
DOG_CEO_BREED_MAP = {
    "beagle.jpg":           "hound-beagle",
    "boxer.jpg":            "boxer",
    "chihuahua.jpg":        "chihuahua",
    "chow.jpg":             "chow",
    "collie_border.jpg":    "collie/border",
    "doberman.jpg":         "doberman",
    "keeshond.jpg":         "keeshond",
    "labrador.jpg":         "retriever/labrador",
    "lhasa.jpg":            "lhasa",
    "malamute.jpg":         "malamute",
    "newfoundland.jpg":     "newfoundland",
    "pekinese.jpg":         "pekinese",
    "pomeranian.jpg":       "pomeranian",
    "rottweiler.jpg":       "rottweiler",
    "samoyed.jpg":          "samoyed",
    "vizsla.jpg":           "vizsla",
    "whippet.jpg":          "whippet",
    "papillon.jpg":         "papillon",
    "pug.jpg":              "pug",
    "dalmatian.jpg":        "dalmatian",
    "random_1.jpg":         "akita",
    "random_2.jpg":         " Norfolk terrier",
    "random_3.jpg":         "chesapeake",
    "random_4.jpg":         "norwegian elkhound",
    "random_5.jpg":         "pinscher/miniature",
}


def download_test_images():
    """Download test images from Dog CEO API if not cached locally."""
    os.makedirs(TEST_DIR, exist_ok=True)

    for entry in TEST_IMAGES:
        filename = entry["file"]
        local_path = os.path.join(TEST_DIR, filename)
        if os.path.exists(local_path):
            continue

        breed_path = DOG_CEO_BREED_MAP.get(filename)
        if not breed_path:
            print(f"  No download mapping for {filename}, skipping")
            continue

        url = f"https://dog.ceo/api/breed/{breed_path}/images/random"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            img_url = resp.json()["message"]
            img_resp = requests.get(img_url, timeout=15)
            img_resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(img_resp.content)
            print(f"  Downloaded {filename}")
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")

    cached = os.listdir(TEST_DIR)
    print(f"  Cached images: {len(cached)}")


def call_predict(api_url, image_path, use_tta=False):
    """Call the /predict endpoint and return the response."""
    url = f"{api_url}/predict"
    if use_tta:
        url += "?tta=true"

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(url, files=files, timeout=30)

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return resp.json()


def run_evaluation(api_url, use_tta_flag):
    """Run the evaluation and return results dict."""
    results = []

    for entry in TEST_IMAGES:
        filename = entry["file"]
        expected = entry["expected"]
        known = entry["known"]

        image_path = os.path.join(TEST_DIR, filename)
        if not os.path.exists(image_path):
            print(f"  SKIP {filename} (not downloaded)")
            continue

        for tta in ([False, True] if use_tta_flag else [False]):
            t_start = time.time()
            response = call_predict(api_url, image_path, use_tta=tta)
            elapsed = time.time() - t_start

            if "error" in response:
                print(f"  ERROR {filename} (TTA={tta}): {response['error']}")
                result = {
                    "file": filename,
                    "expected": expected,
                    "known": known,
                    "tta": tta,
                    "error": response["error"],
                }
            else:
                primary = response.get("primary", {})
                top_k = response.get("top_k", [])
                is_unknown = response.get("is_unknown", False)

                predicted_breed = primary.get("breed", "error")
                confidence = primary.get("confidence", 0)
                correct = (not is_unknown) and (predicted_breed == expected) if known else None

                result = {
                    "file": filename,
                    "expected": expected,
                    "known": known,
                    "tta": tta,
                    "predicted": predicted_breed,
                    "confidence": round(confidence, 4),
                    "is_unknown": is_unknown,
                    "correct": correct,
                    "top_k": top_k,
                    "latency_ms": round(elapsed * 1000, 1),
                }

            results.append(result)
            status = ""
            if correct is True:
                status = " CORRECT"
            elif correct is False:
                status = " WRONG"
            elif known and is_unknown:
                status = " MISSED (unknown)"
            tta_str = "TTA" if tta else "   "
            print(f"  [{tta_str}] {filename:30s} → {result.get('predicted', 'ERROR'):30s} "
                  f"conf={result.get('confidence', 0):.4f}{status}")

    return results


def print_summary(results):
    """Print a summary table matching TESTING.md format."""
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for tta_label, tta_val in [("Baseline (no TTA)", False), ("With TTA", True)]:
        subset = [r for r in results if r.get("tta") == tta_val and "error" not in r]
        if not subset:
            continue

        known = [r for r in subset if r["known"]]
        unknown = [r for r in subset if not r["known"]]
        correct = [r for r in known if r.get("correct") is True]
        wrong = [r for r in known if r.get("correct") is False]
        confidences = [r["confidence"] for r in subset if not r.get("is_unknown")]
        low_conf = [r for r in subset if r["confidence"] < 0.60]
        high_conf = [r for r in subset if r["confidence"] > 0.90]

        print(f"\n--- {tta_label} ---")
        print(f"  Accuracy (known):    {len(correct)}/{len(known)} = {len(correct)/max(len(known),1):.1%}")
        print(f"  Avg confidence:     {sum(confidences)/max(len(confidences),1):.1%}")
        if confidences:
            sorted_c = sorted(confidences)
            print(f"  Median confidence:   {sorted_c[len(sorted_c)//2]:.1%}")
        print(f"  Low conf (<60%):     {len(low_conf)}/{len(subset)} ({len(low_conf)/max(len(subset),1):.0%})")
        print(f"  High conf (>90%):    {len(high_conf)}/{len(subset)} ({len(high_conf)/max(len(subset),1):.0%})")

        if wrong:
            print(f"\n  Misclassified ({len(wrong)}):")
            for r in wrong:
                top3_str = ", ".join(
                    f"{t['breed']} ({t['confidence']:.1%})" for t in r.get("top_k", [])[:3]
                )
                print(f"    {r['file']:30s} expected={r['expected']:25s} "
                      f"got={r['predicted']:25s} conf={r['confidence']:.1%}")
                print(f"      top-3: {top3_str}")

        if unknown:
            print(f"\n  Unknown breeds ({len(unknown)}):")
            for r in unknown:
                top3_str = ", ".join(
                    f"{t['breed']} ({t['confidence']:.1%})" for t in r.get("top_k", [])[:3]
                )
                print(f"    {r['file']:30s} predicted={r['predicted']:25s} "
                      f"conf={r['confidence']:.1%} unknown={r['is_unknown']}")
                print(f"      top-3: {top3_str}")

    # Specifically flag previously-failing cases
    print(f"\n--- Previously Failing Cases ---")
    for filename in ["papillon.jpg", "pug.jpg", "dalmatian.jpg", "random_1.jpg"]:
        matching = [r for r in results if r["file"] == filename and not r.get("tta")]
        if matching:
            r = matching[0]
            status = "FIXED" if r.get("correct") else ("UNKNOWN (OK)" if r.get("is_unknown") else "STILL FAILING")
            print(f"  {filename:30s} → {r.get('predicted', 'ERROR'):25s} "
                  f"conf={r.get('confidence', 0):.1%}  [{status}]")


def save_results(results):
    """Save results to a timestamped JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dog Vision API")
    parser.add_argument("--api_url", default=API_URL, help=f"API base URL (default: {API_URL})")
    parser.add_argument("--no-tta", action="store_true", help="Skip TTA evaluation")
    parser.add_argument("--download", action="store_true", help="Only download test images, don't evaluate")
    args = parser.parse_args()

    print("=" * 60)
    print("DOG VISION — EVALUATION")
    print(f"  API: {args.api_url}")
    print(f"  TTA: {'disabled' if args.no_tta else 'enabled'}")
    print("=" * 60)

    print("\n1. Ensuring test images are cached...")
    download_test_images()

    if args.download:
        print("\nDownload complete. Run without --download to evaluate.")
        return

    print(f"\n2. Checking API health...")
    try:
        resp = requests.get(args.api_url, timeout=10)
        if resp.status_code != 200:
            print(f"  API not healthy: HTTP {resp.status_code}")
            sys.exit(1)
        info = resp.json()
        print(f"  API version: {info.get('version', '?')}")
    except requests.ConnectionError:
        print(f"  Cannot connect to {args.api_url}. Start the server:")
        print(f"    uvicorn app.main:app --reload")
        sys.exit(1)

    print(f"\n3. Running predictions...")
    results = run_evaluation(args.api_url, use_tta_flag=not args.no_tta)

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
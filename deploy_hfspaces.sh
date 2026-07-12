#!/usr/bin/env bash
set -euo pipefail

HF_USERNAME="mozzamshahid"
SPACE_NAME="dog-vision"
SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
SPACE_GIT="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo "=========================================="
echo "  Dog Vision → Hugging Face Spaces Deploy"
echo "=========================================="
echo ""
echo "  HF User:   ${HF_USERNAME}"
echo "  Space:     ${SPACE_NAME}"
echo "  Space URL: ${SPACE_URL}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="/tmp/hf-${SPACE_NAME}-deploy"

if [ -d "${TMP_DIR}" ]; then
  echo "Cleaning up old temp dir..."
  rm -rf "${TMP_DIR}"
fi

echo "Cloning HF Space repo..."
git clone "${SPACE_GIT}" "${TMP_DIR}"
cd "${TMP_DIR}"

git lfs install

echo "Setting up Git LFS for model files..."
if [ ! -f .gitattributes ]; then
  git lfs track "*.keras"
  git lfs track "*.h5"
fi

echo "Creating HF Spaces README.md metadata..."
cat > README.md << 'METADATA'
---
title: Dog Vision
emoji: 🐕
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Dog Vision — Live Dog Breed Detection

Point your camera at a dog and see its breed in real time. Supports 120 breeds with bounding boxes around each dog.
METADATA

echo "Copying project files..."
cp -r "${SCRIPT_DIR}/app" .
cp -r "${SCRIPT_DIR}/data" .
cp -r "${SCRIPT_DIR}/models" .
cp "${SCRIPT_DIR}/Dockerfile" .
cp "${SCRIPT_DIR}/.dockerignore" .
cp "${SCRIPT_DIR}/requirements.txt" .
cp "${SCRIPT_DIR}/runtime.txt" . 2>/dev/null || true

echo "Staging files..."
git add -A

echo ""
echo "Ready to commit and push."
echo ""
echo "Files staged:"
git status --short
echo ""
echo "To deploy now, run:"
echo "  cd ${TMP_DIR}"
echo "  git commit -m 'Deploy Dog Vision to HF Spaces'"
echo "  git push"
echo ""
echo "Then visit: ${SPACE_URL}"
echo ""
echo "Or run all at once:"
read -p "Commit and push now? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  git commit -m "Deploy Dog Vision to HF Spaces"
  git push
  echo ""
  echo "Pushed! Build will take ~10-15 min."
  echo "Check status at: ${SPACE_URL}"
else
  echo "Skipped. You can do it manually from ${TMP_DIR}"
fi
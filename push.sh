#!/bin/bash

# ── 1. Validate commit message ──────────────────────────────
if [ -z "$1" ]; then
  echo "❌ Error: Please provide a commit message."
  echo "Usage: ./push.sh \"Your commit message here\""
  exit 1
fi

# ── 2. Check & zip model files if needed ────────────────────
MODEL_DIR="artifacts"
PKL1="$MODEL_DIR/models.pkl"
PKL2="$MODEL_DIR/vectorizer.pkl"
ZIPFILE="$MODEL_DIR/models.zip"

if [[ -f "$PKL1" && -f "$PKL2" ]]; then
  echo "📦 Zipping $PKL1 and $PKL2 to $ZIPFILE..."
  zip -j "$ZIPFILE" "$PKL1" "$PKL2" && \
  echo "🧹 Removing unzipped files..." && \
  rm -f "$PKL1" "$PKL2"
else
  echo "ℹ️ No unzipped model files found — skipping zip step."
fi

# ── 3. Git push logic ───────────────────────────────────────
git add .
git commit -m "$1"
git push origin main

#!/usr/bin/env bash

# ─────────────────────────────────────────────────────────────
# 📊 Predict on user-provided ground truth CSV (inside Docker)
# Usage: ./eval_custom.sh path/to/your.csv
# ─────────────────────────────────────────────────────────────

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "❌ Usage: $0 path/to/your_csv_file.csv"
    exit 1
fi

INPUT_CSV="$1"

# Check if file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "❌ File not found: $INPUT_CSV"
    exit 1
fi

echo "📦 Copying $INPUT_CSV into Docker container volume..."
cp "$INPUT_CSV" data/tmp_ground_truth.csv

echo "🐳 Running prediction inside Docker..."
docker compose run web python web/diet_classifiers.py \
    --ground_truth /app/data/tmp_ground_truth.csv

echo "✅ Done. Check logs and ground_truth_predictions.csv for output."

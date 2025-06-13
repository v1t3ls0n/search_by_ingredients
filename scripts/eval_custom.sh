# ============================================
# eval_custom.sh - Windows-compatible version
# ============================================

#!/usr/bin/env bash

# ─────────────────────────────────────────────────────────────
# 📊 Predict on user-provided CSV using pre-trained models
# Usage: ./eval_custom.sh path/to/your.csv [--force]
# 
# Note: Assumes models are already trained
# Add --force flag to recompute embeddings for new data
# ─────────────────────────────────────────────────────────────

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "❌ Usage: $0 path/to/your_csv_file.csv [--force]"
    exit 1
fi

INPUT_CSV="$1"
FORCE_FLAG=""

# Check for --force flag
if [ "${2:-}" = "--force" ]; then
    FORCE_FLAG="--force"
    echo "🔄 Force flag detected - will recompute embeddings for new data"
fi

# Check if file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "❌ File not found: $INPUT_CSV"
    exit 1
fi

# Ensure data directory exists
mkdir -p data

echo "📦 Copying $INPUT_CSV into Docker container volume..."
cp "$INPUT_CSV" data/tmp_ground_truth.csv

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to initialize..."
sleep 5

# Check if models exist
echo "🔍 Checking for pre-trained models..."
if docker-compose exec web test -f artifacts/models.pkl; then
    echo "✅ Found pre-trained models"
else
    echo "⚠️  No pre-trained models found. Training models first..."
    docker-compose exec web python3 web/diet_classifiers.py --train --mode both
fi

echo "🐳 Running prediction on custom CSV..."
docker-compose exec web python3 web/diet_classifiers.py \
    --ground_truth data/tmp_ground_truth.csv $FORCE_FLAG

echo "✅ Evaluation complete!"
echo "📊 Check logs for results"
echo "📄 Predictions saved to: /app/artifacts/ground_truth_predictions.csv"
echo "💡 Copy results: docker cp \$(docker-compose ps -q web):/app/artifacts/ground_truth_predictions.csv ."
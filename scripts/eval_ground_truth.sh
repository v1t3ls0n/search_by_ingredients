# ============================================
# eval_ground_truth.sh - Windows-compatible version
# ============================================

#!/usr/bin/env bash

# ────────────────────────────────────────────────
# 🚀 Evaluate pre-trained models on gold set via Docker
# By Guy Vitelson
# 
# Note: Assumes models are already trained
# Run train.sh first if models don't exist
# ────────────────────────────────────────────────

set -euo pipefail

echo "📦 Building and launching containers..."
docker-compose up -d --build

echo "⏳ Waiting for containers to be ready..."
sleep 5

# Check if models exist
echo "🔍 Checking for pre-trained models..."
if docker-compose exec web test -f artifacts/models.pkl; then
    echo "✅ Found pre-trained models"
else
    echo "⚠️  No pre-trained models found. Run train.sh first or use run_full_pipeline.sh"
    echo "❓ Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🧠 Running evaluation on gold set..."
docker-compose exec web python3 web/diet_classifiers.py \
    --ground_truth data/ground_truth_sample.csv --mode text

echo "✅ Evaluation complete!"
echo "📊 Check logs above for results"
echo "📄 Predictions saved to: /app/artifacts/ground_truth_predictions.csv"

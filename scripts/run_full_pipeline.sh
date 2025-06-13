#!/usr/bin/env bash

# ───────────────────────────────────────────────────────────────
# 🎯 End-to-End ML Pipeline Runner (inside Docker)
# By Guy Vitelson
#
# - Builds Docker containers
# - Trains models on silver labels
# - Evaluates on gold-standard CSV
# - Classifies a test ingredient list
#
# Usage:
#   ./run_full_pipeline.sh
# ───────────────────────────────────────────────────────────────

set -euo pipefail

echo "🛠️  Building Docker images..."
docker compose build

echo "🐳 Launching container to run the pipeline..."
docker compose run web bash -c '
    set -euo pipefail
    echo "📦 Ensuring Python deps are present (image should already have them)..."
    pip install -r requirements.txt --quiet || true

    echo "🧠 Training and evaluating text and image classifiers on 100% of the data (sample_frac=1.0)…"
    python web/diet_classifiers.py --train --mode both --sample_frac 1.0

    echo "🧪 Evaluating on provided gold set..."
    python web/diet_classifiers.py --ground_truth /app/data/ground_truth_sample.csv

    echo "🥘 Classifying custom ingredient list..."
    python web/diet_classifiers.py --ingredients "almond flour, erythritol, egg whites"
'

echo "✅  All done!"
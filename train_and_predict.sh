#!/usr/bin/env bash
set -euo pipefail

echo "🛠️  Building Docker images..."
docker compose build

echo "🐳 Launching a one-off container to run the pipeline..."
docker compose run --rm web bash -c '
    set -euo pipefail
    echo "📦 Ensuring Python deps are present (image should already have them)..."
    pip install -r requirements.txt --quiet || true

    echo "🧠 Training and evaluating text and image classifiers on 100% of the data (sample_frac=1.0)…"
    python web/diet_classifiers.py --train --mode both --sample_frac 1.0


    echo "🧪 Evaluating on provided gold set..."
    python web/diet_classifiers.py --ground_truth /app/data/ground_truth_sample.csv

    # Classify custom ingredient list
    python diet_classifiers.py --ingredients "almond flour, erythritol, egg whites"
'

echo "✅  All done!"
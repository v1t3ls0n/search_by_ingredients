#!/usr/bin/env bash
set -euo pipefail

echo "🛠️  Building Docker images..."
docker compose build

echo "🐳 Launching a one-off container to run the pipeline..."
docker compose run --rm web bash -c '
    set -euo pipefail
    echo "📦 Ensuring Python deps are present (image should already have them)..."
    pip install -r requirements.txt --quiet || true

    echo "🧠 Training & testing on both image and text classifiers (sample 10%)..."
    python web/diet_classifiers.py --train --mode both

    echo "🧪 Evaluating on provided gold set..."
    python web/diet_classifiers.py --ground_truth /app/data/ground_truth_sample.csv
'

echo "✅  All done!"

#!/bin/bash

set -e

start_time=$(date +%s)

echo "🛠️  [1/5] Building Docker containers..."
docker compose build

echo "🚀 [2/5] Starting containers in detached mode..."
docker compose up -d

echo "⏳ [3/5] Waiting for 'web' service to be ready..."
sleep 10  # You can adjust this or add a healthcheck poll if needed

echo "🐳 [4/5] Entering 'web' container and running full training + evaluation..."
docker compose exec web bash -c "
    echo '📦 Installing Python dependencies...'
    pip install -r requirements.txt >/dev/null 2>&1 || true

    echo '🧠 [TRAINING] Training with both image and text features (image first)...'
    python web/diet_classifiers.py --train --mode both --sample_frac 0.0135

    echo '🧪 [EVALUATION] Evaluating on full gold set...'
    python web/diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv
"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "✅ [5/5] Done! Total time: ${elapsed}s"

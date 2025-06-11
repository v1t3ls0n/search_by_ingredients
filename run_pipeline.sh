#!/bin/bash

set -e

echo "🛠️ Building Docker containers..."
docker compose build

echo "🚀 Starting containers in detached mode..."
docker compose up -d

echo "⏳ Waiting for 'web' service to be ready..."
sleep 10

echo "🐳 Entering container and running pipeline on ground truth..."
docker compose exec web bash -c "
    echo '📦 Installing dependencies (if needed)...'
    pip install -r requirements.txt >/dev/null 2>&1 || true

    echo '🧠 Training and testing using both image and text classifiers...'
    python web/diet_classifiers.py --train --mode both --sample_frac 0.025

    echo '🧪 Evaluating on provided gold set...'
    python web/diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv
"

echo "✅ Done!"

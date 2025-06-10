#!/bin/bash

set -e

echo "🛠️ Building Docker containers..."
docker compose build

echo "🚀 Starting containers in detached mode..."
docker compose up -d

echo "⏳ Waiting for 'web' service to be ready..."
sleep 10  # adjust if needed depending on hardware

echo "🐳 Entering 'web' container and running full training + evaluation..."
docker compose exec web bash -c "
    echo '📦 Installing dependencies...'
    pip install -r requirements.txt >/dev/null 2>&1 || true

    echo '🧠 Training and testing on silver set (both text + image)...'
    python web/diet_classifiers.py --train --mode both

    echo '🧪 Evaluating on provided gold set...'
    python web/diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv
"

echo "✅ Done!"

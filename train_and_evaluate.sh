#!/bin/bash

set -e

echo "🧠 Training with both text and image features..."
docker compose exec web bash -c "
    python web/diet_classifiers.py --train --mode both
"

echo "🧪 Evaluating on ground truth set..."
docker compose exec web bash -c "
    python web/diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv
"

echo "✅ Training and evaluation complete!"

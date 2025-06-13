#!/usr/bin/env bash

# ───────────────────────────────────────────────────────
# 🧠 Train text + image models on silver-labeled dataset
# ───────────────────────────────────────────────────────

set -euo pipefail

echo "🛠️  Building Docker images..."
docker compose build

echo "🚀 Training classifiers..."
docker compose run --rm web \
    python web/diet_classifiers.py --train --mode both --sample_frac 1.0

echo "✅ Training complete. Models saved to /app/artifacts."

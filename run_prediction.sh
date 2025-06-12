#!/bin/bash

# ────────────────────────────────────────────────
# 🚀 Run end-to-end prediction on gold set via Docker
# By Guy Vitelson
# ────────────────────────────────────────────────

set -e

echo "📦 Building and launching containers..."
docker-compose up -d --build

echo "⏳ Waiting for containers to be ready..."
sleep 10  # Adjust if your services need more time to boot

# Optional: Show container status
docker ps | grep v1t3ls0n

echo "🧠 Running prediction on gold set (ground_truth)..."
docker exec -it search_by_ingredients_v1t3ls0n-web-1 \
    python3 web/diet_classifiers.py --ground_truth /usr/src/data/gold_sample.csv

echo "✅ Done. Check logs above for prediction results."

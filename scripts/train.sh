#!/usr/bin/env bash
# ============================================
# train.sh - Windows-compatible version
# ============================================

# ───────────────────────────────────────────────────────
# 🧠 Train text + image models on silver-labeled dataset
# ───────────────────────────────────────────────────────

set -euo pipefail

echo "🔧 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 5

echo "🎯 Starting training pipeline..."
echo "📝 Note: This will use cached embeddings if available, or compute them fresh if not."

# Use relative path since WORKDIR is /app
docker-compose exec web python3 web/diet_classifiers.py --train --mode both

echo "✅ Training complete!"
echo "📊 Models saved to:"
echo "   - /app/artifacts/models.pkl"
echo "   - /app/artifacts/vectorizer.pkl"
echo "   - /app/artifacts/best_params.json"
echo ""
echo "💡 Tip: To force fresh embeddings, add --force flag to the training command"
echo "💡 Tip: View logs with: docker-compose logs -f web"

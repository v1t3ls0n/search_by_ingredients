# ============================================
# run_full_pipeline.sh - Windows-compatible version
# ============================================

#!/usr/bin/env bash

# ───────────────────────────────────────────────────────────────
# 🎯 End-to-End ML Pipeline Runner (inside Docker)
# By Guy Vitelson
#
# - Builds Docker containers
# - Trains models on silver labels (uses cached embeddings if available)
# - Evaluates on gold-standard CSV
# - Classifies a test ingredient list
#
# Usage:
#   ./run_full_pipeline.sh [--force]  # --force to recompute embeddings
# ───────────────────────────────────────────────────────────────

set -euo pipefail

# Check for --force flag
FORCE_FLAG=""
if [ "${1:-}" = "--force" ]; then
    FORCE_FLAG="--force"
    echo "🔄 Force flag detected - will recompute embeddings"
fi

echo "🛠️  Shutting Down existing containers..."
docker-compose down

echo "🛠️  Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 5

echo "🐳 Running full pipeline..."
docker-compose exec web bash -c "
    set -euo pipefail
    
    echo '📦 Verifying Python dependencies...'
    pip install -r requirements.txt --quiet || true

    echo '🧠 Training classifiers on 100% of the data...'
    echo '📝 Note: Will use cached embeddings if available (add --force to recompute)'
    python3 web/diet_classifiers.py --train --mode both --sample_frac 0.1 $FORCE_FLAG

    echo '🧪 Evaluating on provided gold set...'
    python3 web/diet_classifiers.py --ground_truth data/ground_truth_sample.csv

    echo '🥘 Classifying custom ingredient list...'
    python3 web/diet_classifiers.py --ingredients 'almond flour, erythritol, egg whites'
"

echo "✅ Pipeline complete!"
echo "📊 Results saved to /app/artifacts/"
echo "💡 View logs: docker-compose logs -f web"

#!/bin/bash
set -euo pipefail

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# -------------------------------
# 1) NLTK setup (idempotent)
# -------------------------------
log "Checking NLTK data..."
python - <<'PY' || true
try:
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
        print('NLTK wordnet data already present')
    except LookupError:
        print('Downloading NLTK wordnet data...')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger')
        print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: Could not setup NLTK: {e}')
    print('Continuing without lemmatization support...')
PY

# -------------------------------
# 2) Wait for OpenSearch to be ready
# -------------------------------
OPENSEARCH_URL="${OPENSEARCH_URL:-http://os:9200}"
log "Waiting for OpenSearch at ${OPENSEARCH_URL} ..."
for i in $(seq 1 60); do
  if curl -fsS "${OPENSEARCH_URL%/}/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null; then
    log "OpenSearch is up."
    break
  fi
  if [ "$i" -eq 60 ]; then
    log "ERROR: OpenSearch not reachable at ${OPENSEARCH_URL}"
  fi
  sleep 1
done

# -------------------------------
# 3) Initial indexing (idempotent)
#    - indexer reads OPENSEARCH_URL and RECIPES_INDEX from env
#    - optional DATA_FILE override (default parquet path)
# -------------------------------
if [ ! -f /app/.indexed ]; then
  log "Running initial indexing..."
  # Index Allrecipes parquet
  python web/index_data.py --force --data_file data/allrecipes.parquet
  # Index Allrecipes GROUND TRUTH sample CSV
  python web/index_data.py --force --data_file data/ground_truth_sample.csv
  # Index new Kaggle dataset
  python web/index_data.py --data_file data/food-ingredients-and-recipe-dataset-with-image/dataset.csv
  # Force re-create indices & alias; indexer uses env OPENSEARCH_URL
  touch /app/.indexed
  log "Indexing completed"
else
  log "Indexing already completed, skipping..."
fi

# -------------------------------
# 4) Start Flask app
# -------------------------------
exec python web/app.py

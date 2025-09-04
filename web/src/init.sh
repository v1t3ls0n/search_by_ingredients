#!/bin/bash
set -euo pipefail

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# 1) NLTK setup (idempotent)
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

# 2) Environment (export so Python sees it)
export OPENSEARCH_URL="${OPENSEARCH_URL:-http://os:9200}"
export RECIPES_INDEX="${RECIPES_INDEX:-recipes_v2}"

# 3) Wait for OpenSearch to be ready
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

# 4) Initial indexing (single call; indexer now handles both datasets)
if [ ! -f /app/.indexed ]; then
  log "Running initial indexing..."
  python web/index_data.py --force
  touch /app/.indexed
  log "Indexing completed"
else
  log "Indexing already completed, skipping..."
fi

# 5) Start Flask app
exec python web/app.py

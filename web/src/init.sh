#!/bin/bash

echo "Checking NLTK data..."
python -c "
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
" || true

# One-time indexing (like the old setup)
if [ ! -f /app/.indexed ]; then
  echo "Running initial indexing..."
  # Allow override via DATA_FILE, but default to the parquet we download in the Dockerfile
  DATA_ARG=""
  if [ -n "${DATA_FILE}" ]; then
    DATA_ARG="--data_file ${DATA_FILE}"
  fi
  python web/index_data.py --opensearch_url "${OPENSEARCH_URL:-http://os:9200}" ${DATA_ARG}
  touch /app/.indexed
  echo "Indexing completed"
else
  echo "Indexing already completed, skipping..."
fi

python web/app.py

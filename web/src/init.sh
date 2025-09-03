#!/bin/bash

# Download NLTK data if not already present
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

# Initial indexing (no --opensearch_url; it’s taken from env by index_data.py)
if [ ! -f /app/.indexed ]; then
  echo "Running initial indexing..."
  python web/index_data.py --force --data_file "/app/data/allrecipes.parquet"
  touch /app/.indexed
  echo "Indexing completed"
else
  echo "Indexing already completed, skipping..."
fi

# Start the Flask application
python web/app.py

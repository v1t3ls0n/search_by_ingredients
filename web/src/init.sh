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

# Check if indexing has been done
if [ ! -f /app/.indexed ]; then
    echo "Running initial indexing..."
    python web/index_data.py --opensearch_url "$OPENSEARCH_URL"
    # Create a marker file to indicate indexing is done
    touch /app/.indexed
    echo "Indexing completed"
else
    echo "Indexing already completed, skipping..."
fi

# Start the Flask application
python web/app.py
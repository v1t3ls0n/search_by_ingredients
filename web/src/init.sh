#!/bin/bash

# Initialize external dependencies (NLTK and USDA)
echo "Initializing external dependencies..."
python web/init_dependencies.py || true

# Check if indexing has been done
if [ ! -f /app/.indexed ]; then
    echo ""
    echo "Running initial indexing..."
    python web/index_data.py --opensearch_url "$OPENSEARCH_URL"
    # Create a marker file to indicate indexing is done
    touch /app/.indexed
    echo "Indexing completed"
else
    echo ""
    echo "Indexing already completed, skipping..."
fi

# Start the Flask application
echo ""
echo "Starting Flask application..."
python web/app.py
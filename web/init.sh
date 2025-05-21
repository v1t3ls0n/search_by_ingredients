#!/bin/bash

# Check if indexing has been done
if [ ! -f /app/.indexed ]; then
    echo "Running initial indexing..."
    python index_data.py
    # Create a marker file to indicate indexing is done
    touch /app/.indexed
    echo "Indexing completed"
else
    echo "Indexing already completed, skipping..."
fi

# Start the Flask application
python app.py 
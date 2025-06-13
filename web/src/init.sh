#!/bin/bash

echo "📦 Checking for models.zip in artifacts..."

if [ -f /app/artifacts/models.zip ]; then
    echo "🔓 Extracting models.zip..."
    unzip -o /app/artifacts/models.zip -d /app/artifacts
    echo "🧹 Deleting models.zip..."
    rm /app/artifacts/models.zip
else
    echo "✅ No models.zip found, skipping extraction."
fi

# Check if indexing has been done
if [ ! -f /app/.indexed ]; then
    echo "📊 Running initial indexing..."
    python web/index_data.py --opensearch_url "$OPENSEARCH_URL"
    touch /app/.indexed
    echo "✅ Indexing completed"
else
    echo "🔁 Indexing already completed, skipping..."
fi

# Start the Flask application
echo "🚀 Launching Flask app..."
python web/app.py

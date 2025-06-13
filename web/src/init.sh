#!/bin/bash

echo "📦 Extract pretrained models (models.zip in pretrained_models)..."

if [ -f /app/pretrained_models/models.zip ]; then
    echo "🔓 Extracting models.zip..."
    unzip -o /app/pretrained_models/models.zip -d /app/pretrained_models
else
    echo "✅ Did not find any pretrained models, skipping extraction."
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

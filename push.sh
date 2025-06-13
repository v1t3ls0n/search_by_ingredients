#!/bin/bash

# Define commit message
COMMIT_MSG="
Unzip models.zip at startup for artifacts recovery
- Updated init.sh to automatically extract models.zip if present in /app/artifacts/
- Ensures models.pkl and vectorizer.pkl are restored at runtime
- Added cleanup step to remove the zip after extraction for cleanliness
- Preserves existing indexing and Flask startup logic
"

# Run git commands
git add .
git commit -m "$COMMIT_MSG"
git push origin main

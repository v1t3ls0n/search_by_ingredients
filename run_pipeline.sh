#!/bin/bash

set -e  # Exit on error
set -x  # Print commands for debug

# Step 1: Build Docker containers
docker compose build

# Step 2: Start containers in the background
docker compose up -d

# Step 3: Run the full training pipeline using both text+image on ground truth
docker compose exec web bash -c "python web/diet_classifiers.py --train --mode both"

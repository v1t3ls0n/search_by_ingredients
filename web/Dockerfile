FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
#    • build-essential → wheels that need a compiler
#    • curl            → file downloads
#    • unzip           → extract USDA zip
#    • dos2unix        → convert scripts in init.sh step
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        unzip \
        dos2unix && \
    rm -rf /var/lib/apt/lists/*

# Create data directory structure
RUN mkdir -p /app/data/usda

# Download Argmax's unlabeled training dataset (Allrecipes parquet)
RUN curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/allrecipes.parquet \
    -o /app/data/allrecipes.parquet

# Download Argmax's labeled ground truth dataset for model evaluation
RUN curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/ground_truth_sample.csv \
    -o /app/data/ground_truth_sample.csv

# Download USDA FoodData Central nutritional database
# This provides authoritative carbohydrate data for keto classification
# and serves as training data augmentation for our weak supervision pipeline
RUN curl -L \
    https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2025-04-24.zip \
    -o /tmp/fdc.zip && \
    unzip -q /tmp/fdc.zip -d /tmp && \
    cp /tmp/FoodData_Central_foundation_food_csv_2025-04-24/food.csv \
       /tmp/FoodData_Central_foundation_food_csv_2025-04-24/food_nutrient.csv \
       /tmp/FoodData_Central_foundation_food_csv_2025-04-24/nutrient.csv \
       /app/data/usda/ && \
    rm -rf /tmp/fdc.zip /tmp/FoodData_Central_foundation_food_csv_2025-04-24

# Install Python dependencies for ML pipeline
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and extract pretrained models if exist
COPY pretrained_models/ /app/pretrained_models/
RUN if [ -f /app/pretrained_models/models.zip ]; then \
        echo "🔓 Extracting pretrained models..." && \
        unzip -o /app/pretrained_models/models.zip -d /app/pretrained_models && \
        rm /app/pretrained_models/models.zip; \
    else \
        echo "✅ No pretrained models found, skipping extraction."; \
    fi

# Configure container networking and startup
EXPOSE 8080

# Initialize application with proper script formatting and permissions
CMD ["sh", "-c", "dos2unix web/init.sh && chmod +x web/init.sh && ./web/init.sh"]
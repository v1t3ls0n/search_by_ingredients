FROM python:3.11-slim

WORKDIR /app

# ──────────────────────────────────────────────────────────────────────────
# 0. System packages
#    • build-essential → wheels that need a compiler
#    • curl            → file downloads
#    • unzip           → extract USDA zip
#    • dos2unix        → convert scripts in init.sh step
# ──────────────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        unzip \
        dos2unix && \
    rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────────────────────────────────
# 1. Data directories
# ──────────────────────────────────────────────────────────────────────────
RUN mkdir -p /app/data/usda

# ──────────────────────────────────────────────────────────────────────────
# 2. Allrecipes parquet  
# ──────────────────────────────────────────────────────────────────────────
RUN curl -L \
    https://argmax.nyc3.digitaloceanspaces.com/recipes/allrecipes.parquet \
    -o /app/data/allrecipes.parquet

# ──────────────────────────────────────────────────────────────────────────
# 3. USDA FDC     
# ──────────────────────────────────────────────────────────────────────────
# Download the foundation-food bundle, extract only the needed CSVs,
# then clean up everything we don’t need.
RUN curl -L \
    https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2025-04-24.zip \
    -o /tmp/fdc.zip && \
    unzip -q /tmp/fdc.zip -d /tmp && \
    cp /tmp/FoodData_Central_foundation_food_csv_2025-04-24/food.csv \
       /tmp/FoodData_Central_foundation_food_csv_2025-04-24/food_nutrient.csv \
       /tmp/FoodData_Central_foundation_food_csv_2025-04-24/nutrient.csv \
       /app/data/usda/ && \
    rm -rf /tmp/fdc.zip /tmp/FoodData_Central_foundation_food_csv_2025-04-24

# ──────────────────────────────────────────────────────────────────────────
# 4. Python deps
# ──────────────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────
# 5. Expose port & run entry-point
# ──────────────────────────────────────────────────────────────────────────
EXPOSE 8080

CMD ["sh", "-c", "dos2unix web/init.sh && chmod +x web/init.sh && ./web/init.sh"]

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Create a clean data directory
RUN mkdir -p /app/data && rm -rf /app/data/ground_truth_sample.csv

# Download the datasets
RUN curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/allrecipes.parquet -o /app/data/allrecipes.parquet \
 && curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/ground_truth_sample.csv -o /app/data/ground_truth_sample.csv

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose app port
EXPOSE 8080

# Entry point script
CMD ["sh", "-c", "dos2unix web/init.sh && chmod +x web/init.sh && ./web/init.sh"]

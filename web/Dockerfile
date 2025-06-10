FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Create data directory
RUN mkdir -p /app/data

# Download the parquet (silver) and gold CSV data
RUN curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/allrecipes.parquet -o /app/data/allrecipes.parquet \
 && curl -L https://argmax.nyc3.digitaloceanspaces.com/recipes/ground_truth_sample.csv -o /app/data/ground_truth_sample.csv

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose app port
EXPOSE 8080

# Run init script inside the container
CMD ["sh", "-c", "dos2unix web/init.sh && chmod +x web/init.sh && ./web/init.sh"]

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
 && rm -rf /var/lib/apt/lists/*

# Data dir
RUN mkdir -p /app/data

# Python deps (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gdown

# ---- Google Drive datasets (REQUIRED) ----
# Provide via build args (compose passes them from .env)
ARG DATA_FILE_ID
ARG DATA_FILE_NAME=allrecipes.parquet
ARG GT_FILE_ID
ARG GT_FILE_NAME=ground_truth_sample.csv

# Fail fast if IDs are missing
RUN test -n "$DATA_FILE_ID" || (echo "ERROR: DATA_FILE_ID is required" && exit 1)
RUN test -n "$GT_FILE_ID"   || (echo "ERROR: GT_FILE_ID is required" && exit 1)

# Download recipes parquet (only gdown)
RUN gdown --id "$DATA_FILE_ID" -O "/app/data/${DATA_FILE_NAME}" \
 && if [ "$DATA_FILE_NAME" != "allrecipes.parquet" ]; then \
      ln -sf "/app/data/${DATA_FILE_NAME}" "/app/data/allrecipes.parquet"; \
    fi

# Download ground-truth CSV (only gdown)
RUN gdown --id "$GT_FILE_ID" -O "/app/data/${GT_FILE_NAME}" \
 && ln -sf "/app/data/${GT_FILE_NAME}" "/app/data/ground_truth.csv" \
 && ls -lh /app/data

# Expose the app port
EXPOSE 8080

# Keep your original entrypoint (run-time CRLF fix + start)
# NOTE: init.sh must be present via your bind mount (e.g., ./web:/app)
CMD ["sh", "-c", "dos2unix web/init.sh && chmod +x web/init.sh && ./web/init.sh"]

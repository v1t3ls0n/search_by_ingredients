# ===== Jupyter notebook image =====
FROM jupyter/base-notebook:latest

USER root
RUN apt-get clean && rm -rf /var/lib/apt/lists/* \
 && apt-get update && apt-get install -y --no-install-recommends \
    curl dos2unix \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt \
 && pip install --no-cache-dir gdown

RUN mkdir -p /usr/src/data && chown -R 1000:100 /usr/src/data

# ---- Google Drive datasets (REQUIRED) ----
ARG NB_DATA_FILE_ID
ARG NB_DATA_FILE_NAME=allrecipes.parquet
ARG NB_GT_FILE_ID
ARG NB_GT_FILE_NAME=ground_truth_sample.csv

# Fail fast if IDs are missing
RUN test -n "$NB_DATA_FILE_ID" || (echo "ERROR: NB_DATA_FILE_ID is required" && exit 1)
RUN test -n "$NB_GT_FILE_ID"   || (echo "ERROR: NB_GT_FILE_ID is required" && exit 1)

# Recipes
RUN gdown --id "$NB_DATA_FILE_ID" -O "/usr/src/data/${NB_DATA_FILE_NAME}" \
 && if [ "$NB_DATA_FILE_NAME" != "allrecipes.parquet" ]; then \
      ln -sf "/usr/src/data/${NB_DATA_FILE_NAME}" "/usr/src/data/allrecipes.parquet"; \
    fi

# Ground truth
RUN gdown --id "$NB_GT_FILE_ID" -O "/usr/src/data/${NB_GT_FILE_NAME}" \
 && ln -sf "/usr/src/data/${NB_GT_FILE_NAME}" "/usr/src/data/ground_truth.csv" \
 && ls -lh /usr/src/data

USER $NB_UID

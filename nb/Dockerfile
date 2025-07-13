FROM jupyter/base-notebook
WORKDIR /usr/src/app

USER root
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create directory as root and set proper ownership
RUN mkdir -p /usr/src/data && \
    chown -R $NB_UID:$NB_GID /usr/src/data

USER $NB_UID
RUN curl -L http://argmax.nyc3.digitaloceanspaces.com/recipes/ground_truth_sample.csv -o /usr/src/data/ground_truth_sample.csv

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Download NLTK data during build with same logic as init.sh
RUN python -c " \
import os; \
try: \
    import nltk; \
    nltk_data_dir = os.path.expanduser('~/.nltk_data'); \
    os.makedirs(nltk_data_dir, exist_ok=True); \
    nltk.data.path.append(nltk_data_dir); \
    try: \
        nltk.data.find('corpora/wordnet'); \
        print('NLTK wordnet data already present') \
    except LookupError: \
        print('Downloading NLTK wordnet data...'); \
        nltk.download('wordnet', download_dir=nltk_data_dir); \
        nltk.download('omw-1.4', download_dir=nltk_data_dir); \
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir); \
        print('NLTK data downloaded successfully') \
except Exception as e: \
    print(f'Warning: Could not setup NLTK: {e}'); \
    print('Continuing without lemmatization support...') \
" || echo "NLTK setup failed, but continuing build..."

ENV JUPYTER_ENABLE_LAB=yes

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
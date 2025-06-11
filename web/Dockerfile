FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first (to cache pip install step)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose app port
EXPOSE 8080

# Entrypoint
CMD ["sh", "-c", "dos2unix web/init.sh && chmod +x web/init.sh && ./web/init.sh"]

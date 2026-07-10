# Use a lightweight python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch to minimize image size and memory usage
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file
COPY requirements.txt .

# Install dependencies (since torch is already installed, sentence-transformers won't download full CUDA torch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port (Render sets PORT env, but we'll specify default 5000)
EXPOSE 5000

# Set environment variables for huggingface and PyTorch to use offline / local cache if needed
ENV HF_HOME=/app/.cache/huggingface

# Pre-download the sentence-transformers model during build time
# This avoids downloading the model on the first request in production (which would cause timeout/lag)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Start the application using gunicorn with 1 worker to save RAM
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "chatbot:app"]

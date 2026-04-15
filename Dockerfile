FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time with optional HF token (Transformers model, not ONNX)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN pip install huggingface_hub && \
    hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir /model --token "$HF_TOKEN"

# Copy application code
COPY voxtral_onnx.py .
COPY server.py .
COPY benchmark.py .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "server.py"]
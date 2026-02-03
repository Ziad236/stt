# ===============================
# Base image (CUDA-ready, CPU-safe)
# ===============================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ===============================
# System deps
# ===============================
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Python setup
# ===============================
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# ===============================
# Working directory
# ===============================
WORKDIR /app

# ===============================
# Python dependencies
# ===============================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy app files
# ===============================
COPY . .

# ===============================
# Make start script executable
# ===============================
RUN chmod +x start.sh

# ===============================
# Environment (safe defaults)
# ===============================
ENV PYTHONUNBUFFERED=1
ENV TORCH_LOGS=0
ENV TORCH_CPP_LOG_LEVEL=ERROR

# ===============================
# Expose STT port
# ===============================
EXPOSE 8001

# ===============================
# Start correctly (BASH, not Python!)
# ===============================
CMD ["bash", "start.sh"]

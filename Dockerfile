# PostQuantumEdge -- portable CPU image for running the simulation/experiments.
#
# This image targets reproducible CPU execution (laptops, CI, x86 servers).
# For the Jetson Orin Nano deployment use deployment/Dockerfile.jetson, which is
# based on the NVIDIA L4T / JetPack runtime and links TensorRT.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs needed by matplotlib / scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU build of PyTorch first (keeps the image small), then the rest.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .
RUN pip install -e .

# Default: run the quick end-to-end pipeline (override as needed).
CMD ["bash", "scripts/run_pipeline.sh", "quick"]

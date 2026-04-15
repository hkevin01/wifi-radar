# ─────────────────────────────────────────────────────────────────────────── #
# WiFi-Radar — Application container                                          #
# Base: python:3.11-slim + ffmpeg (for RTMP push) + OpenGL (for OpenCV)       #
# ─────────────────────────────────────────────────────────────────────────── #
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source
COPY . .

# Pre-create data/log dirs
RUN mkdir -p /root/.wifi_radar/data /app/weights /app/logs

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050')" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--simulation", "--dashboard-port", "8050"]

# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, ffmpeg, and network tools
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ffmpeg \
    ca-certificates \
    dnsutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create required directories (relative to backend at /app/backend)
# - thumbnails: generated event thumbnails
# - input_videos: uploaded/sample videos
# - /tmp/Ultralytics: YOLO model cache
RUN mkdir -p backend/thumbnails \
    /tmp/Ultralytics

# Set proper permissions
# 755 = rwxr-xr-x (owner: read/write/execute, group: read/execute, others: read/execute)
# 777 = rwxrwxrwx (full permissions for all - used for writable directories)
RUN chmod -R 755 backend \
    && chmod -R 777 backend/thumbnails \
    && chmod -R 777 backend/input_videos \
    && chmod -R 777 /tmp/Ultralytics

# Create a non-root user for security (recommended for Hugging Face Spaces)
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /tmp/Ultralytics \
    && chown -R appuser:appuser backend/thumbnails \
    && chown -R appuser:appuser backend/input_videos
USER appuser

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run from backend directory (main.py, config paths are relative to backend)
WORKDIR /app/backend

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

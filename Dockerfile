# Enterprise Dockerfile for TerraPulse Climate Intelligence Platform
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY frontend/ ./frontend/
COPY run_server.py .
COPY main.py .

ENV HOST=0.0.0.0
ENV PORT=8008
EXPOSE 8008

CMD ["python", "run_server.py"]

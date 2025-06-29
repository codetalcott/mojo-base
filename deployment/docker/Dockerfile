# Dockerfile for Mojo Semantic Search
# Production deployment with real portfolio corpus

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Mojo (placeholder - actual installation would use official Mojo installer)
RUN echo "Installing Mojo..." && \
    curl -s https://get.modular.com | sh - && \
    echo "Mojo installation complete"

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/portfolio_corpus.json ./data/
COPY deployment/lambda_deploy/api_server.py ./

# Copy Mojo source files
COPY src/integration/*.mojo ./mojo_src/
COPY src/performance/*.mojo ./mojo_src/
COPY src/corpus/*.mojo ./mojo_src/

# Set environment variables
ENV MOJO_PATH="/root/.modular/pkg/packages.modular.com_mojo"
ENV PATH="$MOJO_PATH/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python3", "api_server.py"]
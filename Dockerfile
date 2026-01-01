# Harakat Arabic Diacritization
# Multi-stage build for minimal image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip wheel

# Install CPU-only PyTorch (smaller image)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    fastapi \
    uvicorn[standard]

# Production stage
FROM python:3.11-slim

LABEL maintainer="Jesse Morgan <jeranaias@gmail.com>"
LABEL description="Harakat: High-accuracy Arabic diacritization (2.29% DER, 99.997% Quran)"
LABEL version="3.5.0"

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY harakat.py .
COPY harakat_v1.py .

# Create non-root user for security
RUN useradd -m -u 1000 harakat
USER harakat

# Expose port for API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from harakat import diacritize; print(diacritize('test'))" || exit 1

# Default command: run as CLI
ENTRYPOINT ["python", "harakat.py"]

# To run as API server, use:
# docker run -p 8000:8000 harakat --serve

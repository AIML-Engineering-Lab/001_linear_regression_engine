# ─────────────────────────────────────────────────────────────
# Multi-stage build: keeps the production image small (~200 MB)
# Stage 1: Install Python deps in a builder
# Stage 2: Copy only the essentials into the final image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools for packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /build/venv
COPY requirements.txt web/api/requirements.txt ./combined_requirements.txt
RUN pip install --no-cache-dir --target /build/deps \
    -r combined_requirements.txt

# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

# Install libgomp (required by many sklearn builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages/

# Copy project source
COPY src/            ./src/
COPY web/api/        ./web/api/
COPY data/           ./data/

# Copy trained model if it exists (optional at build time)
COPY --chown=nonroot:nonroot models/ ./models/

# Create a non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.pkl

CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

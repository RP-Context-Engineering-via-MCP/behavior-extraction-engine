# ============================================================
# Stage 1: dependency builder
# ============================================================
# Using a full image here lets pip compile any C-extensions.
# The compiled wheels are copied into the lean runtime stage.
FROM python:3.11-slim AS builder

# Build-time system deps (gcc, libpq-dev for psycopg binary wheel)
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      libpq-dev \
      postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first — maximises Docker layer cache.
# The expensive `pip install` layer is only re-run when requirements.txt changes.
COPY requirements.txt .

RUN pip install --upgrade pip \
    # Install PyTorch CPU-only wheel into /install first to avoid the multi-GB CUDA variant.
    # --prefix=/install is required so torch ends up in the same prefix that is
    # copied into the runtime stage; without it torch is invisible at runtime.
    && pip install --prefix=/install --no-cache-dir \
        torch==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu


# ============================================================
# Stage 2: runtime image
# ============================================================
FROM python:3.11-slim AS runtime

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq5 \
      postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security best practice
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Pull installed packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source (respects .dockerignore)
COPY api/         ./api/
COPY config/      ./config/
COPY db/          ./db/
COPY models/      ./models/
COPY services/    ./services/
COPY utils/       ./utils/
COPY app.py       .

# ----------------------------------------------------------------
# Pre-download the sentence-transformers embedding model so it is
# baked into the image — avoids network calls and cold-start latency
# at runtime.  The cache is stored inside /app so the non-root user
# already has ownership after the chown below.
# ----------------------------------------------------------------
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

RUN python -c \
      "from sentence_transformers import SentenceTransformer; \
       SentenceTransformer('all-MiniLM-L6-v2')"

# Hand ownership to the non-root user
RUN chown -R appuser:appgroup /app

USER appuser

# Expose the port uvicorn listens on (default 8000, override with PORT env var)
EXPOSE 8001

# Health check — Docker / Kubernetes will use this to know the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8001}/health')" || exit 1

# Entrypoint:
#   --workers 1        single worker — scale horizontally with replicas, not threads
#   --no-access-log    uvicorn's own access log; we handle this in middleware
# Environment variable overrides are supported: PORT, LOG_LEVEL
CMD ["sh", "-c", \
     "uvicorn app:app \
        --host 0.0.0.0 \
        --port ${PORT:-8001} \
        --workers 1 \
        --log-level ${LOG_LEVEL:-info} \
        --no-access-log"]

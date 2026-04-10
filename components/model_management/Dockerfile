# =============================================================================
# IMPORTANT: Build context must be the 'components/' directory.
# Run:  docker build -f model_management/Dockerfile -t mmc .   (from components/)
# =============================================================================

# Stage 0: Validate build context - ensures 'common/' is accessible
FROM busybox AS context_check
COPY common/pyproject.toml /tmp/context_validated

FROM python:3.11-slim AS base

ARG COMMITHASH

WORKDIR /app/model_management

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Validate build context (this COPY forces the context_check stage to run)
COPY --from=context_check /tmp/context_validated /tmp/.context_ok

# Copy common package first (needed for local dependency)
COPY common /app/common

# Copy project metadata
COPY model_management/pyproject.toml model_management/uv.lock ./

# Install dependencies from lockfile
RUN uv sync --no-dev

# Copy source
COPY model_management/src ./src

# Set Python path
ENV PYTHONPATH=/app/model_management/src
ENV PATH="/app/model_management/.venv/bin:$PATH"
ENV MARQO_HOST=0.0.0.0
RUN echo $COMMITHASH > build_info.txt

# Run the FastAPI app with uvicorn
CMD ["sh", "-c", "exec uvicorn model_management.main:app --host ${MARQO_HOST} --port 8883"]

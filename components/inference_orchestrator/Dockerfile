# =============================================================================
# IMPORTANT: Build context must be the 'components/' directory.
# Run:  docker build -f inference_orchestrator/Dockerfile -t mioc .   (from components/)
# =============================================================================

# Stage 0: Validate build context - ensures 'common/' is accessible
FROM busybox AS context_check
COPY common/pyproject.toml /tmp/context_validated

FROM python:3.11-slim AS base

ARG COMMITHASH

WORKDIR /app/inference_orchestrator

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
COPY inference_orchestrator/pyproject.toml inference_orchestrator/uv.lock ./

# Install dependencies from lockfile
RUN uv sync --no-dev

# Download NLTK data
RUN uv run python -c "import nltk; nltk.download('punkt_tab')"

# Copy source
COPY inference_orchestrator/src ./src

# Set Python path
ENV PYTHONPATH=/app/inference_orchestrator/src
ENV PATH="/app/inference_orchestrator/.venv/bin:$PATH"
RUN echo $COMMITHASH > build_info.txt

ENV MARQO_HOST=0.0.0.0
ENV MARQO_TIMEOUT_KEEP_ALIVE=75
ENV MARQO_API_WORKERS=1


CMD ["sh", "-c", "uvicorn inference_orchestrator.main:app --host ${MARQO_HOST} \
--timeout-keep-alive ${MARQO_TIMEOUT_KEEP_ALIVE} --workers ${MARQO_API_WORKERS} --port 8884"]
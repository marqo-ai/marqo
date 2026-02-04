FROM python:3.11-slim AS base

ARG COMMITHASH

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project metadata
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile
RUN uv sync --no-dev

# Download NLTK data
RUN uv run python -c "import nltk; nltk.download('punkt_tab')"

# Copy sourced
COPY src ./src

# Set Python path
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"
RUN echo $COMMITHASH > build_info.txt

ENV MARQO_HOST=0.0.0.0
ENV MARQO_TIMEOUT_KEEP_ALIVE=75
ENV MARQO_API_WORKERS=1


CMD ["sh", "-c", "uvicorn inference_orchestrator.main:app --host ${MARQO_HOST} \
--timeout-keep-alive ${MARQO_TIMEOUT_KEEP_ALIVE} --workers ${MARQO_API_WORKERS} --port 8884"]
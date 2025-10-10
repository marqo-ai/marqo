FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/mlib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project metadata
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile
RUN uv sync --no-dev

# Copy sourced
COPY src ./src

# Set Python path
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Download NLTK data
RUN uv run python -m nltk.downloader punkt

ENV MARQO_HOST=0.0.0.0

CMD ["sh", "-c", "uvicorn inference_orchestrator.main:app --host ${MARQO_HOST} --port 8884"]
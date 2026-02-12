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

# Copy source
COPY src ./src

# Set Python path
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"
ENV MARQO_HOST=0.0.0.0
RUN echo $COMMITHASH > build_info.txt

# Run the FastAPI app with uvicorn
CMD ["sh", "-c", "exec uvicorn model_management.main:app --host ${MARQO_HOST} --port 8883"]

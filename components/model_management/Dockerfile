FROM python:3.11-slim AS base

ARG TARGETPLATFORM
ARG COMMITHASH

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

# Copy source
COPY src ./src

# Set Python path
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"
RUN echo $COMMITHASH > build_info.txt

# Run with gunicorn and uvicorn worker
CMD ["uvicorn", "model_management.main:app", "--host", "0.0.0.0", "--port", "8883"]

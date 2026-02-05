# =============================================================================
# IMPORTANT: Build context must be the 'components/' directory as shared files needed to be included.
# Run:  docker build -f marqo/Dockerfile -t marqo .   (from components/)
# =============================================================================

# Stage 0: Validate build context - ensures 'common/' is accessible
FROM busybox AS context_check
COPY common/pyproject.toml /tmp/context_validated

# Stage 1: Build the Java package using Maven
FROM maven:3.8.7-openjdk-18-slim AS maven_build

WORKDIR /app/vespa
# Copy only the pom.xml and any other files required for dependency resolution
COPY marqo/vespa/pom.xml /app/vespa/
# Download dependencies (this layer will be cached if pom.xml hasn't changed)
RUN mvn dependency:go-offline

COPY marqo/vespa/src /app/vespa/src
# Enable parallel builds with Maven (-T 1C uses one thread per CPU core)
RUN mvn clean package -T 1C -X

ARG COMMITHASH

FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

FROM base

WORKDIR /app/marqo

# Validate build context (this COPY forces the context_check stage to run)
COPY --from=context_check /tmp/context_validated /tmp/.context_ok

COPY --from=maven_build /app/vespa/target/marqo-custom-searchers-deploy.jar /app/marqo/vespa/target/

# Copy common package first (needed for local dependency)
COPY common /app/common

# Copy project metadata
COPY marqo/pyproject.toml marqo/uv.lock ./

# Install dependencies from lockfile
RUN uv sync --no-dev

# Install nltk data
RUN uv run python -c "import nltk; nltk.download('punkt_tab')"

COPY marqo/run_marqo.sh ./run_marqo.sh
COPY marqo/src ./src

# Set Python path
ENV PYTHONPATH=/app/marqo/src
ENV PATH="/app/marqo/.venv/bin:$PATH"

RUN chmod +x ./run_marqo.sh
RUN echo $COMMITHASH > build_info.txt
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]
# Stage 1: Build the Java package using Maven
FROM maven:3.8.7-openjdk-18-slim AS maven_build

WORKDIR /app/vespa
# Copy only the pom.xml and any other files required for dependency resolution
COPY vespa/pom.xml /app/vespa/
# Download dependencies (this layer will be cached if pom.xml hasn't changed)
RUN mvn dependency:go-offline

COPY vespa/src /app/vespa/src
# Enable parallel builds with Maven (-T 1C uses one thread per CPU core)
RUN mvn clean package -T 1C


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

COPY --from=maven_build /app/vespa/target/marqo-custom-searchers-deploy.jar /app/vespa/target/
# Copy project metadata
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile
RUN uv sync --no-dev

# Install nltk data
RUN uv run python -c "import nltk; nltk.download('punkt_tab')"


COPY run_marqo.sh /app/run_marqo.sh
COPY src /app/src
COPY ../../../version app/src/version

# Set Python path
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

RUN chmod +x ./run_marqo.sh
RUN echo $COMMITHASH > build_info.txt
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]
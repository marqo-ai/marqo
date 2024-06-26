# Stage 1: Build the Java package using Maven
FROM maven:3.8.7-openjdk-18-slim as maven_build

WORKDIR /app/scripts/vespa_local
COPY scripts/vespa_local/pom.xml .
COPY scripts/vespa_local/src ./src

# Run Maven clean and package
RUN mvn clean package

# Stage 2: Base image for Python setup
FROM marqoai/marqo-base:20 as base_image

# Allow mounting volume containing data and configs for vespa
VOLUME /opt/vespa/var
# Allow mounting volume to expose vespa logs
VOLUME /opt/vespa/logs
# This is required when mounting var folder from an older version of vespa (>30 minor version gap)
# See https://docs.vespa.ai/en/operations-selfhosted/live-upgrade.html for details
ENV VESPA_SKIP_UPGRADE_CHECK true

ARG TARGETPLATFORM
ARG COMMITHASH
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY scripts scripts

# Stage 3: Final stage that builds on the base image
FROM base_image
COPY . /app

# Copy the artifacts from the Maven build stage
COPY --from=maven_build /app/scripts/vespa_local/target /app/scripts/vespa_local/target

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN chmod +x ./run_marqo.sh
RUN echo $COMMITHASH > build_info.txt
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]

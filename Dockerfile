# Stage 1: Build the Java package using Maven
FROM maven:3.8.7-openjdk-18-slim as maven_build

WORKDIR /app/vespa
COPY vespa .
RUN mvn clean package

# Stage 2: Base image for Python setup
FROM marqoai/marqo-base:30 as base_image

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
RUN rm requirements.txt

# Stage 3: Final stage that builds on the base image
FROM base_image

COPY --from=maven_build /app/vespa/target/marqo-custom-searchers-deploy.jar /app/vespa/target/
COPY scripts/ /app/scripts
COPY run_marqo.sh /app/run_marqo.sh
COPY src /app/src


ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN chmod +x ./run_marqo.sh
RUN echo $COMMITHASH > build_info.txt
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]


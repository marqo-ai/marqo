FROM quay.io/almalinux/almalinux:8 AS base_image

ARG TARGETPLATFORM

# Update the public key
RUN rpm --import https://repo.almalinux.org/almalinux/RPM-GPG-KEY-AlmaLinux-8

# Install base packages that are used across both the application and Vespa
RUN dnf install -y epel-release dnf-utils ca-certificates curl gnupg && \
    dnf config-manager --set-enabled powertools

# Install application specific packages
RUN dnf install -y \
        lsof \
        java-17-openjdk \
        python39 \
        python39-devel \
        gcc \
        jq \
        unzip \
        tmux

# Set up Python 3.9 and pip
RUN alternatives --set python3 /usr/bin/python3.9 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3

# Install pip dependencies
COPY requirements requirements
# Install requirements based on the architecture
RUN if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
      pip3 install --no-cache-dir -r requirements/arm64-requirements.txt; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
      pip3 install --no-cache-dir -r requirements/amd64-gpu-requirements.txt; \
    else \
      echo "Unsupported platform: ${TARGETARCH}" && exit 1; \
    fi

# Setup scripts and execute them
COPY scripts scripts
RUN bash scripts/install_redis.sh && \
    bash scripts/install_punkt_tokenizers.sh

# Install ffmpeg based on the architecture
RUN if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
      bash /scripts/install_ffmpeg.sh; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
      bash /scripts/install_ffmpeg_cuda.sh;  \
      # Choose the java version
      update-alternatives --set java /usr/lib/jvm/java-17-openjdk-17.0.13.0.11-3.el8.x86_64/bin/java; \
    else \
      echo "Unsupported platform: ${TARGETARCH}" && exit 1; \
    fi

# Install Vespa and pin the version. All versions can be found using `dns list vespa`
# This is installed as a separate docker layer since we need to upgrade vespa regularly
RUN dnf config-manager --add-repo https://raw.githubusercontent.com/vespa-engine/vespa/master/dist/vespa-engine.repo && \
    dnf install -y vespa-8.431.32-1.el8

ADD scripts/start_vespa.sh /usr/local/bin/start_vespa.sh

# Set Envs for Vespa
ENV PATH="/opt/vespa/bin:/opt/vespa-deps/bin:${PATH}"
# TODO check if following env vars are required
ENV VESPA_LOG_STDOUT="true"
ENV VESPA_LOG_FORMAT="vespa"
ENV VESPA_CLI_HOME=/tmp/.vespa
ENV VESPA_CLI_CACHE_DIR=/tmp/.cache/vespa
ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute,video
# expose nltk data to all users
ENV NLTK_DATA=/root/nltk_data


# Stage 1: Build the Java package using Maven
FROM maven:3.8.7-openjdk-18-slim as maven_build

WORKDIR /app/vespa
COPY vespa .
RUN mvn clean package

# Stage 2: Base image for Python setup
FROM base_image as base_python_image

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
FROM base_python_image

COPY --from=maven_build /app/vespa/target/marqo-custom-searchers-deploy.jar /app/vespa/target/
COPY scripts/ /app/scripts
COPY run_marqo.sh /app/run_marqo.sh
COPY src /app/src


ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN chmod +x ./run_marqo.sh
RUN echo $COMMITHASH > build_info.txt
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]


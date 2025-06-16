# Stage 1: Build the Java package using Maven
FROM maven:3.8.7-openjdk-18-slim as maven_build

WORKDIR /app/vespa
# Copy only the pom.xml and any other files required for dependency resolution
COPY vespa/pom.xml /app/vespa/
# Download dependencies (this layer will be cached if pom.xml hasn't changed)
RUN mvn dependency:go-offline

COPY vespa/src /app/vespa/src
# Enable parallel builds with Maven (-T 1C uses one thread per CPU core)
RUN mvn clean package -T 1C

# Stage 2: Base image for Python setup
FROM marqoai/marqo-base:49 as base_image

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

# TODO This is temporary change to install the packages in requirements.txt file, will be moved to base image in the future
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# Stage 3: Final stage that builds on the base image
FROM base_image

USER root

## Safely install NGINX with repo fallbacks
#RUN dnf config-manager --setopt=cuda-rhel8-x86_64.skip_if_unavailable=true --save && \
#    dnf config-manager --setopt=remi-modular.skip_if_unavailable=true --save && \
#    dnf install -y nginx && dnf clean all
#
## Copy NGINX config (place your file in ./nginx/nginx.conf)
#COPY nginx/nginx.conf /etc/nginx/nginx.conf

# Install build dependencies
RUN dnf groupinstall -y "Development Tools" && \
    dnf install -y \
        pcre-devel \
        tar \
        make \
        gcc \
        systemd-devel \
        openssl-devel \
        zlib-devel \
        curl && \
    dnf clean all

# Install HAProxy 2.9.4
RUN curl -O https://www.haproxy.org/download/2.9/src/haproxy-2.9.4.tar.gz && \
    tar xzf haproxy-2.9.4.tar.gz && \
    cd haproxy-2.9.4 && \
    make TARGET=linux-glibc USE_OPENSSL=1 USE_PROMEX=1 USE_ZLIB=1 USE_PCRE=1 USE_SYSTEMD=1 -j$(nproc) && \
    make install && \
    cd .. && rm -rf haproxy-2.9.4 haproxy-2.9.4.tar.gz

COPY haproxy/haproxy.cfg /etc/haproxy/haproxy.cfg



COPY --from=maven_build /app/vespa/target/marqo-custom-searchers-deploy.jar /app/vespa/target/
COPY scripts/ /app/scripts
COPY run_marqo.sh /app/run_marqo.sh
COPY src /app/src

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN chmod +x ./run_marqo.sh
RUN echo $COMMITHASH > build_info.txt
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]


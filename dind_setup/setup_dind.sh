set -e

apt update \
    && apt install -y ca-certificates openssh-client \
    wget curl iptables supervisor \
    && rm -rf /var/lib/apt/list/*

export DOCKER_CHANNEL=stable \
  DOCKER_VERSION=20.10.18 \
	DOCKER_COMPOSE_VERSION=1.29.2 \
	DEBUG=false

set -eux; \
	\
	arch="$(uname --m)"; \
	case "$arch" in \
        # amd64
		x86_64) dockerArch='x86_64' ;; \
        # arm32v6
		armhf) dockerArch='armel' ;; \
        # arm32v7
		armv7) dockerArch='armhf' ;; \
        # arm64v8
		aarch64) dockerArch='aarch64' ;; \
#		*) echo >&2 "error: unsupported architecture ($arch)"; exit 1 ;;\
	esac; \
	\
	if ! wget -O docker.tgz "https://download.docker.com/linux/static/${DOCKER_CHANNEL}/${dockerArch}/docker-${DOCKER_VERSION}.tgz"; then \
		echo >&2 "error: failed to download 'docker-${DOCKER_VERSION}' from '${DOCKER_CHANNEL}' for '${dockerArch}'"; \
		exit 1; \
	fi; \
	\
	tar --extract \
		--file docker.tgz \
		--strip-components 1 \
		--directory /usr/local/bin/ \
	; \
	rm docker.tgz; \
	\
	dockerd --version; \
	docker --version

cp /app/dind_setup/deps/modprobe /usr/local/bin/
cp -r /app/dind_setup/deps/supervisor/ /etc/supervisor/conf.d/
mkdir -p /opt/bash-utils
cp -r /app/dind_setup/deps/logger.sh /opt/bash-utils/logger.sh

chmod +x /usr/local/bin/modprobe

curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
	&& chmod +x /usr/local/bin/docker-compose && docker-compose version



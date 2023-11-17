FROM marqoai/marqo-base:test-open-clip-upgrade as base_image
VOLUME /var/lib/docker
ARG TARGETPLATFORM
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY scripts scripts
RUN bash scripts/dind_setup/setup_dind.sh

FROM base_image
COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN chmod +x ./run_marqo.sh
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]

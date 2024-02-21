FROM marqoai/marqo-base:18 as base_image
VOLUME /opt/vespa/
ARG TARGETPLATFORM
ARG COMMITHASH
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY scripts scripts

FROM base_image
COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN chmod +x ./run_marqo.sh
RUN echo $COMMITHASH > build_info.txt
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]

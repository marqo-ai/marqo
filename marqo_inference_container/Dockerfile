FROM nvcr.io/nvidia/tritonserver:25.05-py3

# Copy model repository into the container
COPY models /models

# (optional) Expose ports for gRPC / HTTP
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

ENTRYPOINT ["/opt/tritonserver/bin/tritonserver", "--model-repository=/models"]
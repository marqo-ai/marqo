# ARG CUDA_VERSION=11.4.3
# FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04 as cuda_image
# FROM ubuntu:20.04 as base_image
FROM marqoai/marqo-base:1 as base_image
VOLUME /var/lib/docker
ARG TARGETPLATFORM
# # this is required for onnx to find cuda
# COPY --from=cuda_image /usr/local/cuda/ /usr/local/cuda/
# WORKDIR /app
# RUN set -x && \
#     apt-get update && \
#     apt-get install ca-certificates curl  gnupg lsof lsb-release jq -y && \
#     apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common -y && \
#     apt-get update && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install python3.8-distutils -y && \
#     # pip is 276 MB!
#     apt-get  install python3.8 python3-pip -y && \
#     # opencv requirements
#     apt-get install ffmpeg libsm6 libxext6 -y && \
#     # Punkt Tokenizer
#     apt-get install unzip -y  && \
#     mkdir -p /root/nltk_data/tokenizers  && \
#     curl https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -o /root/nltk_data/tokenizers/punkt.zip && \
#     unzip /root/nltk_data/tokenizers/punkt.zip  -d /root/nltk_data/tokenizers/ && \
#     echo Target platform is "$TARGETPLATFORM"

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY scripts scripts
RUN set -x && \
#     bash scripts/install_onnx_gpu_for_amd.sh && \
#     bash scripts/install_torch_amd.sh && \
#     # redis installation for throttling
#     bash scripts/install_redis.sh && \
#     # redis config lines
#     echo "echo never > /sys/kernel/mm/transparent_hugepage/enabled" >> /etc/rc.local && \
#     echo "save ''" | tee -a /etc/redis/redis.conf && \
    # set up Docker-in-Docker
    bash scripts/dind_setup/setup_dind.sh

# Separate the previous steps
FROM base_image
COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN chmod +x ./run_marqo.sh
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]

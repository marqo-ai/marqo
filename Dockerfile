# DIND:
# docker rm -f marqo; DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqo_docker_0
# DEBUGGING:
# export BUILDKIT_PROGRESS=plain; docker rm -f marqo; DOCKER_BUILDKIT=1 docker build --no-cache . -t marqo_docker_0 && docker run --name marqo --privileged -p 8882:8882 marqo_docker_0
ARG CUDA_VERSION=11.4.2
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04 as cuda_image

FROM ubuntu:20.04
VOLUME /var/lib/docker
ARG TARGETPLATFORM
# this is required for onnx to find cuda
COPY --from=cuda_image /usr/local/cuda/ /usr/local/cuda/
WORKDIR /app
RUN apt-get update
RUN apt-get install ca-certificates curl  gnupg lsof lsb-release jq -y
RUN apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common -y
RUN apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.8-distutils -y # python3-distutils
RUN apt-get  install python3.8 python3-pip -y # pip is 276 MB!
# TODO: up the RAM
RUN pip3 --no-cache-dir install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --upgrade
RUN echo Target platform is "$TARGETPLATFORM"

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN if [[ "$TARGETPLATFORM" != "linux/arm64" ]] ; then pip3 --no-cache-dir install --upgrade onnxruntime-gpu ; else pip3 --no-cache-dir install onnxruntime; fi
COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN chmod +x ./run_marqo.sh
CMD ./run_marqo.sh
ENTRYPOINT ["bash", "run_marqo.sh"]
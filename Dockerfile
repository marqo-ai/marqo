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

RUN echo Target platform is "$TARGETPLATFORM"

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY dind_setup dind_setup
RUN bash dind_setup/setup_dind.sh
COPY scripts scripts
RUN bash scripts/install_onnx_gpu_for_amd.sh
COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN chmod +x ./run_marqo.sh
CMD ["./run_marqo.sh"]
ENTRYPOINT ["./run_marqo.sh"]

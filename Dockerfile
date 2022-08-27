# DOOD:
# docker rm -f marqo; DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && docker run --name marqo -v /var/run/docker.sock:/var/run/docker.sock -p 8000:8000 marqo_docker_0
# docker run --name marqo -v /var/run/docker.sock:/var/run/docker.sock -p 8000:8000 marqo_docker_0
# docker run --name opensearch -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:2.1.0
# DIND:
# docker rm -f marqo; DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && docker run --name marqo --privileged -p 8000:8000 --add-host host.docker.internal:host-gateway marqo_docker_0
# DEBUGGING:
# export BUILDKIT_PROGRESS=plain; docker rm -f marqo; DOCKER_BUILDKIT=1 docker build --no-cache . -t marqo_docker_0 && docker run --name marqo --privileged -p 8000:8000 marqo_docker_0

#FROM mreimbold/debian10-dind:latest
FROM cruizba/ubuntu-dind
WORKDIR /app
RUN apt-get update
RUN apt-get install ca-certificates curl  gnupg lsof lsb-release jq -y
RUN apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common -y
RUN apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
#RUN apt-get install python3-distutils-extra -y # python3-distutils
RUN apt-get install python3.8-distutils -y # python3-distutils
RUN apt-get  install python3.8 python3-pip -y # pip is 276 MB!
# TODO: up the RAM

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
#COPY ./src /app/src
COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:/app"
#COPY run_marqo.sh /app/run_marqo.sh
#COPY tox.ini /app/run_marqo.sh
#COPY tests /app/tests
#COPY setup.py /app/setup.py
#COPY setup.py /app/setup.py
RUN chmod +x ./run_marqo.sh
CMD ./run_marqo.sh
ENTRYPOINT ["bash", "run_marqo.sh"]
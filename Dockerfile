# DOCKER_BUILDKIT=1 docker build .

FROM python:3.8-slim-buster
WORKDIR /app
RUN apt-get update
RUN apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y
RUN apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common -y
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update
RUN apt install docker-ce docker-ce-cli containerd.io -y
# RUN apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y
# do we even need to copy across requirements?
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
#COPY ./src /app/src
COPY . /app
#COPY run_marqo.sh /app/run_marqo.sh
#COPY tox.ini /app/run_marqo.sh
#COPY tests /app/tests
#COPY setup.py /app/setup.py
#COPY setup.py /app/setup.py
CMD sh ./run_marqo.sh
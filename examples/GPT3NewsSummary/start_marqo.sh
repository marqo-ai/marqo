#!/bin/bash

docker pull marqoai/marqo:0.0.6;
docker rm -f marqo;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.6
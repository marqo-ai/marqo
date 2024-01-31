#!/bin/bash

docker pull marqoai/marqo:latest;
docker rm -f marqo;
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
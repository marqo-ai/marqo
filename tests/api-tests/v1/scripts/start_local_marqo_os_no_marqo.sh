#!/bin/bash
# Only starts marqo-os. Doesn't start marqo
# this is used for unit testing

export LOCAL_OPENSEARCH_URL="https://localhost:9200"

docker rm -f marqo-os &&
    docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3 &
# wait for marqo-os to start
until [[ $(curl -v --silent --insecure $LOCAL_OPENSEARCH_URL 2>&1 | grep Unauthorized) ]]; do
    sleep 0.1;
done;
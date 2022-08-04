#!/bin/bash

# Start opensearch in the background
if [[ $(docker ps -a | grep 9600 | grep -v Exited) ]]; then
    echo "opennsearch is running"
else
    echo "opensearch not running"
    # this runs docker, using the host's docker
    docker run -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:2.1.0
fi


# Start the neural search web app in the background
cd /app/src/marqo/neural_search || exit
uvicorn api:app --reload &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
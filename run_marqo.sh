#!/bin/bash

# start docker daemon
#dockerd

# Start opensearch in the background
echo doing iffff lsof:
lsof -i:9600
if [[ $(lsof -i:9600) ]]; then
    echo "opennsearch is running"
else
    docker run -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:2.1.0 &
fi



# Start the neural search web app in the background
cd /app/src/marqo/neural_search || exit
uvicorn api:app --reload &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
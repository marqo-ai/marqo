#!/bin/bash

# Start opensearch in the background
docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:2.1.0 &

# Start the neural search web app in the background
cd /app/src/marqo/neural_search || exit
uvicorn api:app --reload &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
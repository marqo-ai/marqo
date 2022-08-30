### Run the Marqo application locally (outside of docker):

1. `cd` into `src/marqo/neuralsearch`
2. Run the following command:
```bash
# if you are running OpenSearch locally, or add an external URL 
export OPENSEARCH_URL="https://localhost:9200" && 
    export PYTHONPATH="${PYTHONPATH}:<your path here>/marqo/src" &&
    uvicorn api:app --host 0.0.0.0 --port 8882 --reload
```

### Build and run the Marqo as a Docker container, that creates and manages its own internal OpenSearch 
1. `cd` into the marqo root directory
2. Run the following command:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqo_docker_0
```

### Build and run the Marqo as a Docker container, connecting to OpenSearch which is running on the host:
1. `cd` into the marqo root directory
2. Run the following command:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway \
         -e "OPENSEARCH_URL=https://localhost:9200" marqo_docker_0
```
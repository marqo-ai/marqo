### Run the Marqo application locally (outside of docker):

1. `cd` into `src/marqo/neuralsearch`
2. Run the following command:
```bash
export OPENSEARCH_URL="https://localhost:9200" && # if you are running OpenSearch locally, or add an external URL 
    export PYTHONPATH="${PYTHONPATH}:<your path here>/marqo/src" &&
    uvicorn api:app --host 0.0.0.0 --reload
```

### Build and run the Marqo as a Docker container, that creates and manages its own internal OpenSearch 
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --privileged -p 8000:8000 --add-host host.docker.internal:host-gateway marqo_docker_0
```

### Build and run the Marqo as a Docker container, connecting to OpenSearch which is running on the host:
```bash
docker rm -f marqo &&
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && 
     docker run --name marqo --privileged -p 8000:8000 --add-host host.docker.internal:host-gateway \
         -e "OPENSEARCH_URL=https://localhost:9200" marqo_docker_0
#!/bin/bash
# args:
# $1 : from_version - the version of the Marqo container you want to start
# $2 : marqo_image_name - name of the Marqo Docker image (default is marqoai/marqo)
# $@ : other_env_vars - additional environment variables to pass to docker

FROM_VERSION="$1"
MARQO_IMAGE="marqoai/marqo"  # Default to 'marqoai/marqo' if not specified
shift 2

# Volume name for Vespa state
VESPA_VAR_VOLUME="opt_vespa_var"

# Function to create and start the container for version 2.9 and later
run_post_2_9_container() {
  docker volume create --name "$VESPA_VAR_VOLUME"
  docker run -d --name marqo -it -p 8882:8882 \
    -e MARQO_ENABLE_BATCH_APIS=TRUE \
    -e "MARQO_MAX_CPU_MODEL_MEMORY=1.6" \
    -v "$VESPA_VAR_VOLUME:/opt/vespa/var" \
    ${@:+"$@"} "$MARQO_IMAGE:$FROM_VERSION"
}

# Function to create and start the container for pre-2.9 versions
run_pre_2_9_container() {
  docker volume create --name "$VESPA_VAR_VOLUME"
  docker run -d --name marqo -it -p 8882:8882 \
    -e MARQO_ENABLE_BATCH_APIS=TRUE \
    -e "MARQO_MAX_CPU_MODEL_MEMORY=1.6" \
    -v "$VESPA_VAR_VOLUME:/opt/vespa" \
    ${@:+"$@"} "$MARQO_IMAGE:$FROM_VERSION"
}

# Check the version and run the appropriate command
if [[ "$FROM_VERSION" =~ ^([0-1]\.|2\.[0-8]) ]]; then
  echo "Starting Marqo container for version pre-2.9..."
  run_pre_2_9_container
else
  echo "Starting Marqo container for version 2.9 and later..."
  run_post_2_9_container
fi

# Follow docker logs (since it is detached)
docker logs -f marqo &
LOGS_PID=$!

# Wait for Marqo to start
until [[ $(curl -v --silent --insecure http://localhost:8882 2>&1 | grep Marqo) ]]; do
  sleep 0.1
done

# Kill the `docker logs` command
kill $LOGS_PID

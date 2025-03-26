#!/bin/bash
# args:
# $1 : marqo_image_name - name of the image you want to test
# $@ : env_vars - strings representing all args to pass to docker-compose
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MARQO_DOCKER_IMAGE="$1"
shift

# Clean up any existing containers
docker compose -f $SCRIPT_DIR/../docker-compose.yml --profile cuda down 2>/dev/null || true

# Create a temporary .env file to store all passed environment variables
TEMP_ENV_FILE=$(mktemp)
echo "# Generated environment file from start_cuda_docker_marqo_split.sh" > $TEMP_ENV_FILE

# Add the required environment variables to the env file
echo "MARQO_DOCKER_IMAGE=$MARQO_DOCKER_IMAGE" >> $TEMP_ENV_FILE
echo "ENV_FILE=$TEMP_ENV_FILE" >> $TEMP_ENV_FILE

# Process all additional arguments as environment variables and add them to the temp file
# This replicates the behavior of ${@:+"$@"} in the docker run command
for arg in "$@"; do
  # Extract variable name and value from each argument
  if [[ $arg == -e* ]]; then
    # Handle -e VAR=VALUE format
    env_var="${arg#-e }"
    echo "$env_var" >> $TEMP_ENV_FILE
  elif [[ $arg == --env=* ]]; then
    # Handle --env=VAR=VALUE format
    env_var="${arg#--env=}"
    echo "$env_var" >> $TEMP_ENV_FILE
  fi
done

# We don't need to set the default CUDA-specific environment variables
# since they are already defined in the docker-compose.yml file

# Debug - print the .env file contents
echo "Contents of $TEMP_ENV_FILE:"
cat $TEMP_ENV_FILE

set -x
# Start the containers using docker-compose with the cuda profile
# Use --env-file to pass the environment variables
docker compose --env-file $TEMP_ENV_FILE -f $SCRIPT_DIR/../docker-compose.yml up -d
set +x

# Follow docker logs (since it is detached)
# Also use --env-file for logs
docker compose --env-file $TEMP_ENV_FILE -f $SCRIPT_DIR/../docker-compose.yml logs -f marqo-cuda &
LOGS_PID=$!

# Wait for marqo to start
until [[ $(curl -v --silent --insecure http://localhost:8882 2>&1 | grep Marqo) ]]; do
    sleep 0.1;
done;

# Kill the `docker logs` command (so subprocess does not wait for it)
kill $LOGS_PID
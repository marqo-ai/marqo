#!/bin/bash
#source /opt/bash-utils/logger.sh
export PYTHONPATH="${PYTHONPATH}:/app/src/"
export CUDA_HOME=/usr/local/cuda/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

trap "bash /app/scripts/shutdown.sh; exit" SIGTERM SIGINT

function wait_for_process () {
    local max_retries=30
    local n_restarts_before_sigkill=3
    local process_name="$1"
    local retries=0
    while ! [[ $(docker ps -a | grep CONTAINER) ]] >/dev/null && ((retries < max_retries)); do
        echo "Process $process_name is not running yet. Retrying in 1 seconds"
        echo "Retry $retries of a maximum of $max_retries retries"
        ((retries=retries+1))
        if ((retries >= n_restarts_before_sigkill)); then
            echo "sending SIGKILL to dockerd and restarting "
            ps axf | grep docker | grep -v grep | awk '{print "kill -9 " $1}' | sh; rm /var/run/docker.pid; dockerd &
        else
            dockerd &
        fi
        sleep 3
        if ((retries >= max_retries)); then
            return 1
        fi
    done
    return 0
}


VESPA_IS_INTERNAL=False
# Vespa local run
if ([ -n "$VESPA_QUERY_URL" ] || [ -n "$VESPA_DOCUMENT_URL" ] || [ -n "$VESPA_CONFIG_URL" ]) && \
   ([ -z "$VESPA_QUERY_URL" ] || [ -z "$VESPA_DOCUMENT_URL" ] || [ -z "$VESPA_CONFIG_URL" ]); then
  echo "Error: Partial VESPA environment variables set. Please provide all or none of the VESPA_QUERY_URL, VESPA_DOCUMENT_URL, VESPA_CONFIG_URL."
  exit 1

elif [ -z "$VESPA_QUERY_URL" ] && [ -z "$VESPA_DOCUMENT_URL" ] && [ -z "$VESPA_CONFIG_URL" ]; then
  # Start local vespa
  echo "Running Vespa Locally"
  tmux new-session -d -s vespa "bash /usr/local/bin/start_vespa.sh"

  echo "Waiting for Vespa to start"
  for i in {1..5}; do
      echo -ne "Waiting... $i seconds\r"
      sleep 1
  done
  echo -e "\nDone waiting."

  # Try to deploy the application and branch on the output
  echo "Setting up Marqo local vector search application..."
  END_POINT="http://localhost:19071/application/v2/tenant/default/application/default"
  MAX_RETRIES=10
  RETRY_COUNT=0

  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Make the curl request and capture the output
    RESPONSE=$(curl -s -X GET "$END_POINT")

    # Check for the specific "not found" error response
    if echo "$RESPONSE" | grep -q '"error-code":"NOT_FOUND"'; then
      echo "Marqo does not find an existing index"
      echo "Marqo is deploying the application and waiting for the response from document API to start..."
      # Deploy a dummy application package
      vespa deploy /app/scripts/vespa_dummy_app --wait 300 >/dev/null 2>&1

      until curl -f -X GET http://localhost:8080 >/dev/null 2>&1; do
        echo "  Waiting for Vespa document API to be available..."
        sleep 10
      done
      echo "  Vespa document API is available. Local Vespa setup complete."
      break

    # Check for the "generation" success response
    elif echo "$RESPONSE" | grep -q '"generation":'; then
      echo "Marqo found an existing index. Waiting for the response from document API to start Marqo..."

      until curl -f -X GET http://localhost:8080 >/dev/null 2>&1; do
        echo "  Waiting for Vespa document API to be available..."
        sleep 10
      done
      echo "  Vespa document API is available. Local Vespa setup complete."
      break
    fi
    ((RETRY_COUNT++))
    sleep 5
  done

  if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: Marqo didn't configure local vector . Marqo is still starting but unexpected error may happen."
  fi

  export VESPA_QUERY_URL="http://localhost:8080"
  export VESPA_DOCUMENT_URL="http://localhost:8080"
  export VESPA_CONFIG_URL="http://localhost:19071"
  export VESPA_IS_INTERNAL=True

else
  echo "All VESPA environment variables provided. Skipping local Vespa setup."
fi

# Start up redis
if [ "$MARQO_ENABLE_THROTTLING" != "FALSE" ]; then
    echo "Starting redis-server"
    redis-server /etc/redis/redis.conf &
    echo "Called redis-server command"

    start_time=$(($(date +%s%N)/1000000))
    while true; do
        redis-cli ping &> /dev/null
        if [ $? -eq 0 ]; then
            break
        fi

        current_time=$(($(date +%s%N)/1000000))
        elapsed_time=$(expr $current_time - $start_time)
        if [ $elapsed_time -ge 2000 ]; then
            # Expected start time should be < 30ms in reality.
            echo "redis-server failed to start within 2s. skipping."
            break
        fi
        sleep 0.1
        
    done
    echo "redis-server is now running"

else
    echo "Throttling has been disabled. Skipping redis-server start."
fi

# set the default value to info and convert to lower case
export MARQO_LOG_LEVEL=${MARQO_LOG_LEVEL:-info}
MARQO_LOG_LEVEL=`echo "$MARQO_LOG_LEVEL" | tr '[:upper:]' '[:lower:]'`

# Start the tensor search web app in the background
cd /app/src/marqo/tensor_search || exit
uvicorn api:app --host 0.0.0.0 --port 8882 --timeout-keep-alive 75 --log-level $MARQO_LOG_LEVEL &
api_pid=$!
wait "$api_pid"


# Exit with status of process that exited first
exit $?

#!/bin/bash

# set the default value to info and convert to lower case
export MARQO_LOG_LEVEL=${MARQO_LOG_LEVEL:-info}
MARQO_LOG_LEVEL=`echo "$MARQO_LOG_LEVEL" | tr '[:upper:]' '[:lower:]'`

export PYTHONPATH="${PYTHONPATH}:/app/src/"
export CUDA_HOME=/usr/local/cuda/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

trap "bash /app/scripts/shutdown.sh; exit" SIGTERM SIGINT

if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
  echo "Python packages:"
  pip freeze
fi

function wait_for_process () {
    local max_retries=30
    local n_restarts_before_sigkill=3
    local process_name="$1"
    local retries=0
    while ! [[ $(docker ps -a | grep CONTAINER) ]] >/dev/null && ((retries < max_retries)); do
        if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
          echo "Process $process_name is not running yet. Retrying in 1 seconds"
          echo "Retry $retries of a maximum of $max_retries retries"
        fi
        echo "Waiting for Marqo-OS to start..."
        ((retries=retries+1))
        if ((retries >= n_restarts_before_sigkill)); then
            if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
              echo "sending SIGKILL to dockerd and restarting "
            fi
            ps axf | grep docker | grep -v grep | awk '{print "kill -9 " $1}' | sh; rm /var/run/docker.pid; dockerd > /dev/null 2>&1 &
        else
            if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
              dockerd &
            else
              dockerd > /dev/null 2>&1 &
            fi
        fi
        sleep 3
        if ((retries >= max_retries)); then
            return 1
        fi
    done
    return 0
}

OPENSEARCH_IS_INTERNAL=False
# Start opensearch in the background
if [[ ! $OPENSEARCH_URL ]]; then

  which docker > /dev/null 2>&1
  rc=$?
  if [ $rc != 0 ]; then
      echo "Docker not found. Installing it..."
      bash /app/dind_setup/setup_dind.sh &
      setup_dind_pid=$!
      wait "$setup_dind_pid"
  fi

  if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
    echo "Starting supervisor"
  fi

  /usr/bin/supervisord -n >> /dev/null 2>&1 &

  dockerd > /dev/null 2>&1 &

  if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
    echo "Called dockerd command. Waiting for dockerd to start."
  fi

  processes=(dockerd)
  for process in "${processes[@]}"; do
      wait_for_process "$process"
      if [ $? -ne 0 ]; then
          echo "$process is not running after max time"
          exit 1
      else
          if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
            echo "$process is running"
          fi
      fi
  done
  OPENSEARCH_URL="https://localhost:9200"
  OPENSEARCH_IS_INTERNAL=True
  if [[ $(docker ps -a | grep marqo-os) ]]; then
      if [[ $(docker ps -a | grep marqo-os | grep -v Up) ]]; then
        docker start marqo-os > /dev/null 2>&1 &
        until [[ $(curl -v --silent --insecure $OPENSEARCH_URL 2>&1 | grep Unauthorized) ]]; do
          sleep 0.1;
        done;
        echo "Marqo-OS started"
      fi
      echo "Marqo-OS is running"
  else
      echo "Marqo-OS not found; starting Marqo-OS..."
      if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
        docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3 &
      else
        docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3  > /dev/null 2>&1 &
      fi
      docker start marqo-os &
      until [[ $(curl -v --silent --insecure $OPENSEARCH_URL 2>&1 | grep Unauthorized) ]]; do
        sleep 0.1;
      done;
  fi
fi

export OPENSEARCH_URL
export OPENSEARCH_IS_INTERNAL

# Start up redis
if [ "$MARQO_ENABLE_THROTTLING" != "FALSE" ]; then
    redis-server /etc/redis/redis.conf
    if [ "$MARQO_LOG_LEVEL" = "debug" ]; then
      echo "Called redis-server command"
    fi

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
            # redis-server failed to start:
            echo "Marqo throttling failed to start within 2s. Continuing without throttling."
            break
        fi
        sleep 0.1
        
    done
    # redis server is now running
    echo "Marqo throttling successfully started"

else
    # skip starting Redis
    echo "Marqo throttling has been disabled. Throttling start-up skipped."
fi



# Start the tensor search web app in the background
cd /app/src/marqo/tensor_search || exit
uvicorn api:app --host 0.0.0.0 --port 8882 --timeout-keep-alive 75 --log-level $MARQO_LOG_LEVEL &
api_pid=$!
wait "$api_pid"


# Exit with status of process that exited first
exit $?

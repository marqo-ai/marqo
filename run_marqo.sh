#!/bin/bash
#source /opt/bash-utils/logger.sh
export PYTHONPATH="${PYTHONPATH}:/app/src/"
export CUDA_HOME=/usr/local/cuda/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

trap "bash /app/scripts/shutdown.sh; exit" SIGTERM SIGINT
echo "Python packages:"
pip freeze

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

OPENSEARCH_IS_INTERNAL=False
# Start Vespa in the background
if [[ ! $VESPA_CONFIG_URL ]]; then
  # Start local vespa
  echo "Running Vespa Locally"
  tmux new-session -d -s vespa "bash /usr/local/bin/start_vespa.sh"
  # Start opensearch in the background

  echo "Waiting for Vespa to start"
  for i in {1..5}; do
      echo -ne "Waiting... $i seconds\r"
      sleep 1
  done
  echo -e "\nDone waiting."
  # Deploy a dummy application to Vespa
  tmux new-session -d -s deploy "vespa deploy /app/scripts/vespa_dummy_app --wait 300"
  echo "Waiting for dummy application to be deployed"
  for i in {1..5}; do
      echo -ne "Waiting... $i seconds\r"
      sleep 1
  done
  echo -e "\nDone waiting."
  export VESPA_QUERY_URL="http://localhost:8080"
  export VESPA_DOCUMENT_URL="http://localhost:8080"
  export VESPA_CONFIG_URL="http://localhost:19071"

else
  echo "Found VESPA_CONFIG_URL. Skipping internal Vespa configuration"
fi

# Start up redis
if [ "$MARQO_ENABLE_THROTTLING" != "FALSE" ]; then
    echo "Starting redis-server"
    redis-server /etc/redis/redis.conf
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

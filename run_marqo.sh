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

  echo "Starting supervisor"
  /usr/bin/supervisord -n >> /dev/null 2>&1 &

  dockerd &
  echo "called dockerd command"

  echo "Waiting for processes to be running"
  processes=(dockerd)
  for process in "${processes[@]}"; do
      wait_for_process "$process"
      if [ $? -ne 0 ]; then
          echo "$process is not running after max time"
          exit 1
      else
          echo "$process is running"
      fi
  done
  OPENSEARCH_URL="https://localhost:9200"
  OPENSEARCH_IS_INTERNAL=True
  if [[ $(docker ps -a | grep marqo-os) ]]; then
      if [[ $(docker ps -a | grep marqo-os | grep -v Up) ]]; then
        docker start marqo-os &
        until [[ $(curl -v --silent --insecure $OPENSEARCH_URL 2>&1 | grep Unauthorized) ]]; do
          sleep 0.1;
        done;
        echo "Opensearch started"
      fi
      echo "OpenSearch is running"
  else
      echo "OpenSearch not found; running OpenSearch"
      docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:2.4.0 &
      docker start marqo-os &
      until [[ $(curl -v --silent --insecure $OPENSEARCH_URL 2>&1 | grep Unauthorized) ]]; do
        sleep 0.1;
      done;
  fi
fi

export OPENSEARCH_URL
export OPENSEARCH_IS_INTERNAL
# Start the tensor search web app in the background
cd /app/src/marqo/tensor_search || exit
uvicorn api:app --host 0.0.0.0 --port 8882 &
api_pid=$!
wait "$api_pid"


# Exit with status of process that exited first
exit $?
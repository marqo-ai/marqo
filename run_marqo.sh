#!/bin/bash
#source /opt/bash-utils/logger.sh

export PYTHONPATH="${PYTHONPATH}:/app/src/"
export PYTHONPATH="${PYTHONPATH}:/app/src/"
export CUDA_HOME=/usr/local/cuda/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

function wait_for_process () {
    local max_time_wait=30
    local process_name="$1"
    local waited_sec=0
#    while ! pgrep "$process_name" >/dev/null && ((waited_sec < max_time_wait)); do
    while ! [[ $(docker ps -a | grep CONTAINER) ]] >/dev/null && ((waited_sec < max_time_wait)); do
        echo "Process $process_name is not running yet. Retrying in 1 seconds"
        echo "Waited $waited_sec seconds of $max_time_wait seconds"
        sleep 1
        ((waited_sec=waited_sec+1))
        if ((waited_sec >= max_time_wait)); then
            return 1
        fi
    done
    return 0
}

OPENSEARCH_IS_INTERNAL=False
# Start opensearch in the background
if [[ ! $OPENSEARCH_URL ]]; then
  bash /app/dind_setup/setup_dind.sh

  echo "Starting supervisor"
  /usr/bin/supervisord -n >> /dev/null 2>&1 &

  echo starting dockerd command
  dockerd &
  echo dockerd command complete

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
      docker run --name marqo-os -id -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.2 &
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
uvicorn api:app --host 0.0.0.0 --port 8882

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
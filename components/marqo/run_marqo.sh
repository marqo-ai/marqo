#!/bin/bash

# set the default value to info and convert to lower case
export MARQO_LOG_LEVEL=${MARQO_LOG_LEVEL:-info}
MARQO_LOG_LEVEL=`echo "$MARQO_LOG_LEVEL" | tr '[:upper:]' '[:lower:]'`

# set the default host to 0.0.0.0
export MARQO_HOST=${MARQO_HOST:-"0.0.0.0"}


# set default number of workers to 1
if [ -z "${MARQO_API_WORKERS}" ]; then
  export MARQO_API_WORKERS=1
fi

# Start the Marqo API in the background
cd /marqo/app/src/marqo/tensor_search
uvicorn api:app --host "$MARQO_HOST" --port 8882 --workers $MARQO_API_WORKERS --timeout-keep-alive 75 --log-level "$MARQO_LOG_LEVEL" &

# Capture the PID of the last background process
export api_pid=$!
# Wait for the Uvicorn process to terminate
wait "$api_pid"
# Exit with status of process that exited first
exit $?

#!/bin/bash
# args:
# $1 : marqo_image_name - name of the image you want to test
# $@ : env_vars - strings representing all args to pass docker call

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start single node vespa
python3 "$SCRIPT_DIR/../../../../scripts/vespa_local/vespa_local.py" full-start
MARQO_DOCKER_IMAGE="$1"
shift

docker rm -f marqo 2>/dev/null || true

# Explanation:
# -d detaches docker from process (so subprocess does not wait for it)
# ${@:+"$@"} adds ALL args (past $1) if any exist.

set -x
docker run -d --name marqo -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e MARQO_MAX_CPU_MODEL_MEMORY=1.6 \
    -e MARQO_ENABLE_BATCH_APIS=true \
    -e VESPA_CONFIG_URL="http://host.docker.internal:19071" \
    -e VESPA_DOCUMENT_URL="http://host.docker.internal:8080" \
    -e VESPA_QUERY_URL="http://host.docker.internal:8080" \
    -e ZOOKEEPER_HOSTS="host.docker.internal:2181" \
    -e MARQO_INDEX_DEPLOYMENT_LOCK_TIMEOUT=0 \
    -e MARQO_MODELS_TO_PRELOAD='[]' \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    ${@:+"$@"} "$MARQO_DOCKER_IMAGE" --memory=8g
set +x

# Follow docker logs (since it is detached)
docker logs -f marqo &
LOGS_PID=$!

# wait for marqo to start
until [[ $(curl -v --silent --insecure http://localhost:8882 2>&1 | grep Marqo) ]]; do
    sleep 0.1;
done;

# Kill the `docker logs` command (so subprocess does not wait for it)
kill $LOGS_PID
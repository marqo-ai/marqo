#!/bin/bash
# This script is meant to be run at buildtime. This install marqo-os on docker,
# for amd64
if [[ "$TARGETPLATFORM" != "linux/amd64" ]]; then
  exit
fi
dockerd &
dockerd_pid=$!
wait "$dockerd_pid"
docker pull marqoai/marqo-os:0.0.2

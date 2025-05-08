#!/bin/bash

set -e

MARQO_IMAGE_NAME="$1"

if [ "$MARQO_IMAGE_NAME" = "marqo_docker_0" ]; then
  echo "Building Marqo image as no image identifier was provided"
  DOCKER_BUILDKIT=1 docker build ../../../ -t marqo_docker_0 -q
else
  echo "Pulling Marqo image with provided identifier"
  docker pull "$MARQO_IMAGE_NAME"
fi

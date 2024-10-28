#!/bin/bash

# Variables
IMAGE_TAG=$1 #
ECR_REGISTRY="424082663841.dkr.ecr.us-east-1.amazonaws.com"
IMAGE_REPO="marqo"

# Log in to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 424082663841.dkr.ecr.us-east-1.amazonaws.com

# Pull the Docker image from ECR
image_full_name="$ECR_REGISTRY/$IMAGE_REPO:$IMAGE_TAG"
echo "Pulling image: $image_full_name"
docker pull "$image_full_name"

# Optionally retag the image locally to marqo-ai/marqo
local_tag="marqo-ai/marqo:$IMAGE_TAG"
echo "Retagging image to: $local_tag"
docker tag "$image_full_name" "$local_tag"

# Now you can use the image as "marqo-ai/marqo:$IMAGE_TAG"

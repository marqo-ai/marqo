#!/bin/bash

REPO_PREFIX=$1
SUFFIXES=("api" "inference-orchestrator" "model-management")

create_repo_if_not_exists() {
    local repo_name=$1
    output=$(aws ecr describe-repositories --repository-names ${repo_name} 2>&1)
    if [ $? -ne 0 ]; then
        if echo ${output} | grep -q RepositoryNotFoundException; then
            echo "Creating ${repo_name} ECR Repository"
            aws ecr create-repository --repository-name ${repo_name}
        else
            >&2 echo ${output}
            return 1
        fi
    else
        echo "${repo_name} ECR Repository already exists"
    fi
}

# Validate input
if [ -z "$REPO_PREFIX" ]; then
    echo "Usage: $0 <repo_prefix>"
    echo "Example: $0 marqoai"
    exit 1
fi

# Strip trailing slashes
REPO_PREFIX="${REPO_PREFIX%/}"

# Create all three repos
for suffix in "${SUFFIXES[@]}"; do
    repo_name="${REPO_PREFIX}/${suffix}"
    create_repo_if_not_exists "${repo_name}"
done
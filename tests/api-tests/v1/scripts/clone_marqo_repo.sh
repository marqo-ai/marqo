#!/bin/bash
# $1: the toxinidir, the path to the tox file.
# $2: the name of the marqo branch to test
# shellcheck disable=SC2155
export MARQO_API_TESTS_ROOT=$(pwd)

. "${MARQO_API_TESTS_ROOT}/conf"
export LOCAL_OPENSEARCH_URL="https://localhost:9200"

if [[ $(pwd | grep -v marqo-api-tests) ]] && [[ $(pwd | grep -v runner/work) ]] && [[ $(pwd | grep -v actions-runner/_work) ]];
then
 echo checked pwd, and it does not look the directory is correct
 echo "The current directory is $PWD"
 exit 1
fi

rm -rf "${MARQO_API_TESTS_ROOT}/temp"
mkdir "${MARQO_API_TESTS_ROOT}/temp"
cd "${MARQO_API_TESTS_ROOT}/temp" || exit

git clone https://github.com/marqo-ai/marqo.git

cd "${MARQO_API_TESTS_ROOT}/temp/marqo" || exit
git fetch
git switch "$2"
git pull

DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 -q || exit 1
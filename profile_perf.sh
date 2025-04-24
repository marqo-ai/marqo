#!/usr/bin/env bash

TEST_NAME=${1:-profile}

PYTHONPATH="$(pwd)/src:$PYTHONPATH"
PKG_FILTER="marqo"
GH_BASE="https://github.com/marqo-ai/marqo"
COMMIT="$(git rev-parse HEAD)"

sudo env PYTHONPATH=$PYTHONPATH \
  py-spy record \
  -o ${TEST_NAME}.svg \
  -- ./venv/bin/python ./src/marqo/tensor_search/perf_test_query_result_parsing.py

./src/marqo/tensor_search/annotate_svg.py ${TEST_NAME}.svg ${TEST_NAME}-annotated.svg "$GH_BASE" "$COMMIT" "$PKG_FILTER"

open "${TEST_NAME}-annotated.svg"
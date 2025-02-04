#!/bin/bash

set -e

PY_MARQO_BRANCH="$1"

if [[ "$PY_MARQO_BRANCH" == "marqo" ]]; then
    MQ_PY_MARQO_BRANCH="marqo"
elif [[ -z "$PY_MARQO_BRANCH" ]]; then
    MQ_PY_MARQO_BRANCH="git+https://github.com/marqo-ai/py-marqo.git@mainline"
else
    MQ_PY_MARQO_BRANCH="git+https://github.com/marqo-ai/py-marqo.git@$PY_MARQO_BRANCH"
fi

pip install "$MQ_PY_MARQO_BRANCH"

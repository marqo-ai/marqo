#!/bin/bash

# CUDA Warmup (in the background)
python3 cuda_warmup.py &

# Run your main application
exec "$@"
#!/bin/bash
# This script is meant to be run at buildtime. This is because onnxruntime-gpu
# doesn't install on arm64
if [[ "$TARGETPLATFORM" ]]; then
  echo environment var TARGETPLATFORM is undefined
  exit 1
fi

if [[ "$TARGETPLATFORM" != "linux/arm64" ]]; then
  pip3 --no-cache-dir install --upgrade onnxruntime-gpu
else
  pip3 --no-cache-dir install --upgrade onnxruntime
fi
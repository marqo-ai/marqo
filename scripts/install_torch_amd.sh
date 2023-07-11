#!/bin/bash
# This script is meant to be run at buildtime. This is because onnxruntime-gpu
# doesn't install on arm64
if ! [[ "$TARGETPLATFORM" ]]; then
  echo ERROR environment var TARGETPLATFORM is undefined
  exit 1
fi

if [[ "$TARGETPLATFORM" != "linux/arm64" ]]; then
  pip3 install numpy --pre torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --force-reinstall --index-url https://download.pytorch.org/whl/cu118
fi
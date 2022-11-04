#!/bin/bash
# This script is meant to be run at buildtime. This is because onnxruntime-gpu
# doesn't install on arm64
if ! [[ "$TARGETPLATFORM" ]]; then
  echo ERROR environment var TARGETPLATFORM is undefined
  exit 1
fi

if [[ "$TARGETPLATFORM" != "linux/arm64" ]]; then
  pip3 --no-cache-dir install --upgrade onnxruntime-gpu
  pip3 install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 --upgrade
else
  pip3 --no-cache-dir install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --upgrade
  pip3 --no-cache-dir install --upgrade onnxruntime
fi
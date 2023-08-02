#!/bin/bash
# This script is meant to be run at buildtime. This is because onnxruntime-gpu
# doesn't install on arm64
if ! [[ "$TARGETPLATFORM" ]]; then
  echo ERROR environment var TARGETPLATFORM is undefined
  exit 1
fi

if [[ "$TARGETPLATFORM" != "linux/arm64" ]]; then
  pip3 install --no-cache-dir torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade
fi
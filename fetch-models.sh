#!/usr/bin/env bash
# Download the two ML models (not committed: size + their own licenses).
set -euo pipefail
mkdir -p models
curl -L -o models/seeta_fd_frontal_v1.0.bin \
  https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin
curl -L -o models/w600k_mbf.onnx \
  https://huggingface.co/immich-app/buffalo_s/resolve/main/recognition/model.onnx
ls -la models/

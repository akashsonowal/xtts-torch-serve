#!/bin/bash

torch-model-archiver \
  --model-name xttsv2 \
  --version 1.0 \
  --model-file mar-files/xtts_v2.py \
  --serialized-file mar-files/xtts_artifacts/model.pth \
  --handler mar-files/handler.py \
  --extra-files mar-files/base_tts.py,mar-files/gcs_bucket.py,mar-files/loggercz.py,mar-files/service_account.json,mar-files/xtts_artifacts/config.json,mar-files/xtts_artifacts/vocab.json,mar-files/xtts_artifacts/clipped_first_15_seconds.wav \
  --export-path ./model-store
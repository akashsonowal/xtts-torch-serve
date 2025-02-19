#!/bin/bash
torchserve --start --ncs \
    --ts-config config.properties \
    --model-store model-store \
    --models xttsv2.mar 
#!/usr/bin/env bash
set -euo pipefail
source /home/myd/workspace/resnet50_cbam_reid/bash_script/_experiment_common.sh

ROOT=${PROJECT_DIR}/model/ablation_attention_type
LOG_ROOT=${PROJECT_DIR}/result/logs/ablation_attention_type

launch_training attention_channel "${ROOT}/channel" "${LOG_ROOT}/channel.log" \
    --cbam_layers 2 3 4 --attention_type channel
launch_training attention_spatial "${ROOT}/spatial" "${LOG_ROOT}/spatial.log" \
    --cbam_layers 2 3 4 --attention_type spatial
launch_training attention_cbam "${ROOT}/cbam" "${LOG_ROOT}/cbam.log" \
    --cbam_layers 2 3 4 --attention_type cbam

wait_for_all

#!/usr/bin/env bash
set -euo pipefail
source /home/myd/workspace/resnet50_cbam_reid/bash_script/_experiment_common.sh

ROOT=${PROJECT_DIR}/model/ablation_position
LOG_ROOT=${PROJECT_DIR}/result/logs/ablation_position

launch_training position_layer2 "${ROOT}/layer2" "${LOG_ROOT}/layer2.log" \
    --cbam_layers 2 --attention_type cbam
launch_training position_layer3 "${ROOT}/layer3" "${LOG_ROOT}/layer3.log" \
    --cbam_layers 3 --attention_type cbam
launch_training position_layer4 "${ROOT}/layer4" "${LOG_ROOT}/layer4.log" \
    --cbam_layers 4 --attention_type cbam
launch_training position_layer34 "${ROOT}/layer34" "${LOG_ROOT}/layer34.log" \
    --cbam_layers 3 4 --attention_type cbam
launch_training position_layer234 "${ROOT}/layer234" "${LOG_ROOT}/layer234.log" \
    --cbam_layers 2 3 4 --attention_type cbam

wait_for_all

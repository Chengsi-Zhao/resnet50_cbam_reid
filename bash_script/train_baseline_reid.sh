#!/usr/bin/env bash
set -euo pipefail
source /home/myd/workspace/resnet50_cbam_reid/bash_script/_experiment_common.sh

launch_training \
    baseline_reid \
    "${PROJECT_DIR}/model/baseline_reid_seed1" \
    "${PROJECT_DIR}/result/logs/baseline_reid_seed1.log" \
    --cbam_layers \
    --attention_type cbam \
    --backbone_lr_mult 1.0 \
    --cbam_lr_mult 1.0 \
    --head_lr_mult 1.0 \
    --warmup_epochs 0

wait_for_all

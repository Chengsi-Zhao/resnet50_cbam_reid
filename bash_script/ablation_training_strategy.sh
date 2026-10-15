#!/usr/bin/env bash
set -euo pipefail
source /home/myd/workspace/resnet50_cbam_reid/bash_script/_experiment_common.sh

ROOT=${PROJECT_DIR}/model/ablation_training_strategy
LOG_ROOT=${PROJECT_DIR}/result/logs/ablation_training_strategy

# 控制组：所有参数使用同一学习率，无 warmup。
launch_training strategy_uniform "${ROOT}/uniform_no_warmup" \
    "${LOG_ROOT}/uniform_no_warmup.log" \
    --cbam_layers 2 3 4 --attention_type cbam \
    --backbone_lr_mult 1.0 --cbam_lr_mult 1.0 --head_lr_mult 1.0 \
    --warmup_epochs 0

# 只改变分组学习率：主干为基础学习率的 0.1 倍，CBAM 和头保持 1 倍。
launch_training strategy_grouped "${ROOT}/grouped_no_warmup" \
    "${LOG_ROOT}/grouped_no_warmup.log" \
    --cbam_layers 2 3 4 --attention_type cbam \
    --backbone_lr_mult 0.1 --cbam_lr_mult 1.0 --head_lr_mult 1.0 \
    --warmup_epochs 0

# 在相同分组学习率上比较 5/10 epoch warmup。
launch_training strategy_warmup5 "${ROOT}/grouped_warmup5" \
    "${LOG_ROOT}/grouped_warmup5.log" \
    --cbam_layers 2 3 4 --attention_type cbam \
    --backbone_lr_mult 0.1 --cbam_lr_mult 1.0 --head_lr_mult 1.0 \
    --warmup_epochs 5 --warmup_start_factor 0.1

launch_training strategy_warmup10 "${ROOT}/grouped_warmup10" \
    "${LOG_ROOT}/grouped_warmup10.log" \
    --cbam_layers 2 3 4 --attention_type cbam \
    --backbone_lr_mult 0.1 --cbam_lr_mult 1.0 --head_lr_mult 1.0 \
    --warmup_epochs 10 --warmup_start_factor 0.1

wait_for_all

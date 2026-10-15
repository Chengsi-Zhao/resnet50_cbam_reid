#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/myd/workspace/resnet50_cbam_reid
DATA_DIR=${PROJECT_DIR}/data/Market-1501-v15.09.15
CONDA_BIN=/data/myd/miniconda3/bin/conda
GPU_ID=${GPU_ID:-7}
MAX_EVAL_PARALLEL=${MAX_EVAL_PARALLEL:-4}

if (( MAX_EVAL_PARALLEL < 1 || MAX_EVAL_PARALLEL > 4 )); then
    echo "MAX_EVAL_PARALLEL 必须在 1～4 之间" >&2
    exit 1
fi

EXPERIMENTS=(
    "baseline|baseline_reid_seed1"
    "position_layer2|ablation_position/layer2"
    "position_layer3|ablation_position/layer3"
    "position_layer4|ablation_position/layer4"
    "position_layer34|ablation_position/layer34"
    "position_layer234|ablation_position/layer234"
    "attention_channel|ablation_attention_type/channel"
    "attention_spatial|ablation_attention_type/spatial"
    "attention_cbam|ablation_attention_type/cbam"
    "strategy_uniform|ablation_training_strategy/uniform_no_warmup"
    "strategy_grouped|ablation_training_strategy/grouped_no_warmup"
    "strategy_warmup5|ablation_training_strategy/grouped_warmup5"
    "strategy_warmup10|ablation_training_strategy/grouped_warmup10"
)

evaluate_one() {
    local name=$1
    local model_dir=$2
    local weights=${PROJECT_DIR}/model/${model_dir}/best_model.pth
    local output_dir=${PROJECT_DIR}/result/${model_dir}/eval_best

    if [[ ! -f "${weights}" ]]; then
        echo "缺少权重: ${weights}" >&2
        return 1
    fi
    mkdir -p "${output_dir}"
    echo "[$(date '+%F %T')] 开始评估 ${name}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA_BIN}" run \
        --no-capture-output -n FedPerfix python "${PROJECT_DIR}/script/eval.py" \
        --data_dir "${DATA_DIR}" \
        --weights "${weights}" \
        --output_dir "${output_dir}" \
        --batch_size 256 \
        --num_workers 4 \
        --query_chunk 512 \
        --gpu 0 >"${output_dir}/eval.log" 2>&1
    echo "[$(date '+%F %T')] 完成评估 ${name}"
}

cd "${PROJECT_DIR}"
pids=()
names=()
failed=0

wait_batch() {
    local index
    for ((index = 0; index < ${#pids[@]}; index++)); do
        if ! wait "${pids[index]}"; then
            echo "评估失败: ${names[index]}" >&2
            failed=1
        fi
    done
    pids=()
    names=()
}

for spec in "${EXPERIMENTS[@]}"; do
    name=${spec%%|*}
    model_dir=${spec#*|}
    evaluate_one "${name}" "${model_dir}" &
    pids+=("$!")
    names+=("${name}")
    if (( ${#pids[@]} >= MAX_EVAL_PARALLEL )); then
        wait_batch
    fi
done
if (( ${#pids[@]} > 0 )); then
    wait_batch
fi

exit "${failed}"

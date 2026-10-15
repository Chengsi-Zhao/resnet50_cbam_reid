#!/usr/bin/env bash
set -euo pipefail

# Market-1501：ResNet-50 + CBAM(layer2/3/4) 单卡训练。
# 固定物理 GPU 7；进程内部该卡会显示为 cuda:0。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

PROJECT_DIR=/home/myd/workspace/resnet50_cbam_reid
DATA_DIR=${PROJECT_DIR}/data/Market-1501-v15.09.15
MODEL_CACHE_DIR=/home/myd/data/models
OUTPUT_DIR=${PROJECT_DIR}/model/cbam_l234_seed1
CONDA_BIN=/data/myd/miniconda3/bin/conda

for split in bounding_box_train bounding_box_test query; do
    if [[ ! -d "${DATA_DIR}/${split}" ]]; then
        echo "缺少数据目录: ${DATA_DIR}/${split}" >&2
        exit 1
    fi
done

if [[ ! -f "${MODEL_CACHE_DIR}/checkpoints/resnet50-11ad3fa6.pth" ]]; then
    echo "缺少 ResNet-50 预训练权重，请先下载到 ${MODEL_CACHE_DIR}/checkpoints" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

"${CONDA_BIN}" run --no-capture-output -n FedPerfix python "${PROJECT_DIR}/script/train.py" \
    --data_dir "${DATA_DIR}" \
    --cbam_layers 2 3 4 \
    --cbam_ratio 16 \
    --cbam_kernel 7 \
    --per_block 1 \
    --feat_dim 512 \
    --pretrained 1 \
    --model_cache_dir "${MODEL_CACHE_DIR}" \
    --dropout 0.0 \
    --img_h 256 \
    --img_w 128 \
    --epochs 80 \
    --batch_size 64 \
    --pid_batch_size 16 \
    --lr 3.5e-4 \
    --weight_decay 5e-4 \
    --lr_step 40 \
    --id_weight 1.0 \
    --triplet_weight 1.0 \
    --margin 0.3 \
    --num_workers 8 \
    --eval_start 70 \
    --output_dir "${OUTPUT_DIR}" \
    --seed 1

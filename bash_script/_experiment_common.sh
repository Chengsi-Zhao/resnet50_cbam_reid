#!/usr/bin/env bash
# 四个训练入口共享的配置和全局并发控制；本文件只应被 source。

PROJECT_DIR=/home/myd/workspace/resnet50_cbam_reid
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data/Market-1501-v15.09.15}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/home/myd/data/models}
CONDA_BIN=${CONDA_BIN:-/data/myd/miniconda3/bin/conda}
GPU_ID=${GPU_ID:-7}

# 所有入口共用 10 个 flock 槽；四个脚本同时启动也最多运行 10 个训练进程。
REQUESTED_MAX_PARALLEL=${MAX_PARALLEL:-10}
if (( REQUESTED_MAX_PARALLEL < 1 )); then
    echo "MAX_PARALLEL 必须大于 0" >&2
    exit 1
fi
if (( REQUESTED_MAX_PARALLEL > 10 )); then
    MAX_GLOBAL_JOBS=10
else
    MAX_GLOBAL_JOBS=${REQUESTED_MAX_PARALLEL}
fi
SLOT_DIR=/tmp/resnet50_cbam_reid_training_slots
PID_DIR=${PROJECT_DIR}/result/pids

EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-64}
PID_BATCH_SIZE=${PID_BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_START=${EVAL_START:-70}
SEED=${SEED:-1}

for command_name in flock "${CONDA_BIN}"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "找不到命令: ${command_name}" >&2
        exit 1
    fi
done
for split in bounding_box_train bounding_box_test query; do
    if [[ ! -d "${DATA_DIR}/${split}" ]]; then
        echo "缺少数据目录: ${DATA_DIR}/${split}" >&2
        exit 1
    fi
done
if [[ ! -f "${MODEL_CACHE_DIR}/checkpoints/resnet50-11ad3fa6.pth" ]]; then
    echo "缺少预训练权重: ${MODEL_CACHE_DIR}/checkpoints/resnet50-11ad3fa6.pth" >&2
    exit 1
fi
mkdir -p "${SLOT_DIR}" "${PID_DIR}"

EXPERIMENT_PIDS=()
EXPERIMENT_NAMES=()

launch_training() {
    local experiment_name=$1
    local output_dir=$2
    local log_file=$3
    shift 3

    mkdir -p "${output_dir}" "$(dirname "${log_file}")"

    local slot slot_fd pid
    while true; do
        for ((slot = 0; slot < MAX_GLOBAL_JOBS; slot++)); do
            exec {slot_fd}>"${SLOT_DIR}/slot_${slot}.lock"
            if flock -n "${slot_fd}"; then
                (
                    trap 'flock -u "$slot_fd"' EXIT
                    echo "[$(date '+%F %T')] 启动 ${experiment_name}"
                    echo "物理 GPU=${GPU_ID}，全局并发槽=${slot}/${MAX_GLOBAL_JOBS}"
                    export CUDA_DEVICE_ORDER=PCI_BUS_ID
                    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
                    export PYTHONUNBUFFERED=1
                    cd "${PROJECT_DIR}"
                    "${CONDA_BIN}" run --no-capture-output -n FedPerfix \
                        python "${PROJECT_DIR}/script/train.py" \
                        --data_dir "${DATA_DIR}" \
                        --cbam_ratio 16 \
                        --cbam_kernel 7 \
                        --per_block 1 \
                        --feat_dim 512 \
                        --pretrained 1 \
                        --model_cache_dir "${MODEL_CACHE_DIR}" \
                        --dropout 0.0 \
                        --img_h 256 \
                        --img_w 128 \
                        --epochs "${EPOCHS}" \
                        --batch_size "${BATCH_SIZE}" \
                        --pid_batch_size "${PID_BATCH_SIZE}" \
                        --lr 3.5e-4 \
                        --weight_decay 5e-4 \
                        --lr_step 40 \
                        --id_weight 1.0 \
                        --triplet_weight 1.0 \
                        --margin 0.3 \
                        --num_workers "${NUM_WORKERS}" \
                        --eval_start "${EVAL_START}" \
                        --output_dir "${output_dir}" \
                        --seed "${SEED}" \
                        "$@"
                    echo "[$(date '+%F %T')] 完成 ${experiment_name}"
                ) >"${log_file}" 2>&1 &
                pid=$!
                # 后台子进程继承锁 FD；父进程关闭自己的副本。
                exec {slot_fd}>&-
                EXPERIMENT_PIDS+=("${pid}")
                EXPERIMENT_NAMES+=("${experiment_name}")
                printf '%s\n' "${pid}" >"${PID_DIR}/${experiment_name}.pid"
                echo "已提交 ${experiment_name}: PID=${pid}, log=${log_file}"
                return 0
            fi
            exec {slot_fd}>&-
        done
        echo "10 个训练槽均占用，${experiment_name} 等待空闲槽……"
        sleep 5
    done
}

wait_for_all() {
    local index failures=0
    for ((index = 0; index < ${#EXPERIMENT_PIDS[@]}; index++)); do
        if wait "${EXPERIMENT_PIDS[index]}"; then
            echo "完成: ${EXPERIMENT_NAMES[index]}"
        else
            echo "失败: ${EXPERIMENT_NAMES[index]}" >&2
            failures=1
        fi
    done
    return "${failures}"
}

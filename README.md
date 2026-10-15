# ResNet-50 + CBAM 行人重识别（Re-ID）

本项目在 ResNet-50 上按需嵌入 CBAM，并使用 Market-1501 完成 Re-ID 训练、标准检索评测及三组消融实验。训练目标由身份分类损失和 Batch-hard Triplet 损失共同组成，评测指标为 mAP 与 Rank-1/5/10/20。

## 项目目录

```text
resnet50_cbam_reid/
├── README.md                    # 项目入口说明
├── requirements.txt             # Python 依赖
├── data/                        # 数据集
├── model/                       # 训练得到的 best/last 权重
├── result/                      # 指标、日志、曲线、PID 与实验报告
├── doc/                         # README 之外的项目文档
├── script/                      # Python 训练、评测和模型代码
└── bash_script/                 # 训练与批量评测 bash 脚本
```

其中：

- `script/resnet_cbam.py`：ResNet-50、CBAM 插入位置与 Re-ID BNNeck。
- `script/cbam.py`：通道注意力、空间注意力及完整 CBAM。
- `script/datasets.py`：Market-1501 加载与 ID-based 采样。
- `script/losses.py`：ID Cross Entropy + Batch-hard Triplet Loss。
- `script/train.py`：训练入口、分组学习率、warmup 与 checkpoint 保存。
- `script/evaluate.py`：训练时调用的标准 Re-ID 指标实现。
- `script/eval.py`：训练后独立评测入口，支持分块计算以降低内存占用。
- `doc/消融实验.md`：13 个实验的配置、启动方法及结果入口。
- `doc/代码问题与修改说明.md`：旧代码问题、修复方法和当前限制。
- `result/ablation_report/REPORT.md`：完整消融结果与结论。

`/home/myd/data/models` 是 torchvision 的 ImageNet 预训练权重缓存，不属于本项目训练 checkpoint，因此保持在项目目录外。

## 数据目录

默认数据路径为：

```text
data/Market-1501-v15.09.15/
├── bounding_box_train/          # 训练图像
├── bounding_box_test/           # Gallery 候选库
└── query/                       # Query 查询图像
```

加载器也兼容 `bounding_box_query/`、`gallery/`、`test/` 和 `pytorch/` 子目录形式。训练集只使用正身份 ID；标准评测保留 Gallery 中的 pid=0 distractor，并排除 pid=-1 junk。

## 模型设计

`--cbam_layers` 控制注意力嵌入的 ResNet stage：

- `--cbam_layers`：不使用注意力，即 Re-ID 基线。
- `--cbam_layers 4`：只在 layer4 插入。
- `--cbam_layers 3 4`：在 layer3、layer4 插入。
- `--cbam_layers 2 3 4`：在中深层插入。
- `--cbam_layers 1 2 3 4`：四个残差 stage 都插入。

`--per_block 1` 时，CBAM 位于每个 Bottleneck 的 `bn3` 之后、shortcut 相加之前，与原论文的残差块插入方式一致；`--per_block 0` 表示每个 stage 结束后仅插入一次，用于消融。

`--attention_type` 支持 `channel`、`spatial` 和 `cbam`，分别表示仅通道注意力、仅空间注意力和二者串联。

## 直接训练

完整 CBAM 示例：

```bash
cd /home/myd/workspace/resnet50_cbam_reid
python script/train.py \
  --data_dir data/Market-1501-v15.09.15 \
  --cbam_layers 2 3 4 \
  --epochs 80 \
  --batch_size 64 \
  --pid_batch_size 16 \
  --output_dir model/cbam_exp1
```

不带 CBAM 的 Re-ID 基线：

```bash
python script/train.py \
  --data_dir data/Market-1501-v15.09.15 \
  --cbam_layers \
  --epochs 80 \
  --batch_size 64 \
  --pid_batch_size 16 \
  --output_dir model/baseline_exp1
```

训练会保存：

- `best_model.pth`：从 `--eval_start` 起，训练期 mAP 最优的 checkpoint。
- `last_model.pth`：最后一个 epoch 的 checkpoint。

新版 checkpoint 同时保存模型结构、训练配置、类别数、输入尺寸、epoch 和评测指标，便于独立评测自动恢复配置。

## Bash 训练入口

单个预设实验：

```bash
mkdir -p result/logs
nohup bash bash_script/train_cbam_gpu7.sh \
  > result/logs/cbam_l234_seed1.log 2>&1 &
```

Re-ID 基线和三组消融实验：

```bash
mkdir -p result/logs/launchers
nohup bash bash_script/train_baseline_reid.sh \
  > result/logs/launchers/baseline.log 2>&1 &
nohup bash bash_script/ablation_position.sh \
  > result/logs/launchers/position.log 2>&1 &
nohup bash bash_script/ablation_attention_type.sh \
  > result/logs/launchers/attention_type.log 2>&1 &
nohup bash bash_script/ablation_training_strategy.sh \
  > result/logs/launchers/training_strategy.log 2>&1 &
```

四个消融入口共用全局文件锁，训练进程总数最多为 10。默认使用物理 GPU 7；可在启动前通过 `GPU_ID`、`MAX_PARALLEL`、`EPOCHS` 等环境变量覆盖公共配置。

## 独立评测

```bash
python script/eval.py \
  --data_dir data/Market-1501-v15.09.15 \
  --weights model/baseline_reid_seed1/best_model.pth \
  --output_dir result/baseline_reid_seed1/eval_best
```

批量评测 13 个最佳模型：

```bash
nohup bash bash_script/evaluate_all_models.sh \
  > result/logs/evaluate_all_models.log 2>&1 &
```

评测脚本会严格加载权重，并把 mAP、Rank-1/5/10/20 以百分数写入对应的 `result/.../eval_best/metrics.txt`。

## 当前实验结论

13 个模型中，仅通道注意力（layer2+3+4）效果最好：mAP 79.7824%，Rank-1 91.7755%。完整 CBAM 的最佳插入位置是 layer3+4；主干 0.1 倍学习率的分组策略在当前配置下明显欠拟合。详细数据和 loss 曲线见 `result/ablation_report/REPORT.md`。

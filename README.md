# ResNet-50 + CBAM 行人重识别(Re-ID)训练代码

用于「面向社区监控视频的行人寻踪算法与系统」项目的**第二步·算法改进**。
核心思路：在基线 ResNet-50 的中深层残差阶段(layer2/3/4)嵌入 CBAM 注意力，
其余数据划分与训练配方保持完全一致(控制变量)，从头训练并对比 mAP / Rank-1。

## 文件说明
| 文件 | 作用 |
|------|------|
| `cbam.py` | CBAM 模块：通道注意力 + 空间注意力(串行) |
| `resnet_cbam.py` | ResNet-50 骨干，CBAM 可配置嵌入 layer2/3/4；含 Re-ID embedding 头与特征提取接口 |
| `datasets.py` | Market-1501 风格数据集加载 + ID-based 采样(满足 triplet loss) |
| `losses.py` | 身份交叉熵(ID loss) + batch-hard Triplet loss |
| `evaluate.py` | 训练内置评估模块：被 `train.py` 在每个 Epoch 末调用，算 Rank-1/5/10 与 mAP |
| `eval.py` | **独立命令行评估脚本**：单独加载某个 `.pth` 权重，用省内存方式补算 mAP / Rank(推荐用于训练后补指标) |
| `train.py` | 训练入口 |

> `evaluate.py` 与 `eval.py` 的区别：前者是训练循环内部调用的评估模块，后者是你**训练结束后单独跑**的命令行入口，二者算法口径一致，但 `eval.py` 专为「16G 内存、评估易 OOM」环境做了省内存设计，补指标时优先用它。

## CBAM 嵌入不同位置(消融实验)
模型把嵌入位置做成了参数 `--cbam_layers`：
- `--cbam_layers 2 3 4`  中深层全部嵌入(改进版基础配置)
- `--cbam_layers 3 4`    只嵌深层
- `--cbam_layers 4`      只嵌 stage4
- `--cbam_layers`  (空)  纯基线 ResNet-50(对照组)

`--per_block` 控制是"每个残差块后都嵌"(1) 还是"每个 stage 末尾嵌一次"(0)。

## 训练示例
改进版(中深层全嵌)：
```
python train.py --data_dir /data/market1501 --cbam_layers 2 3 4 \
    --epochs 60 --batch_size 32 --lr 3.5e-4 --output_dir runs/cbam_exp1
```
基线(不加 CBAM，其余参数一模一样)：
```
python train.py --data_dir /data/market1501 --cbam_layers \
    --epochs 60 --batch_size 32 --lr 3.5e-4 --output_dir runs/baseline
```

> 内存紧张时可加 `--num_workers 0 --eval_start 1000`：先跳过训练内置的全量评估，让模型一路跑完 60 个 Epoch 把 `last_model.pth` 存好，**训练结束后再用 `eval.py` 单独补算 mAP**(见下文「独立评估 eval.py」)。

在 GPU 节点(gpu1)后台长跑，防止断连：
```
nohup python train.py --data_dir /data/market1501 --cbam_layers 2 3 4 \
    --output_dir runs/cbam_exp1 > train_cbam.log 2>&1 &
tail -f train_cbam.log
```

## 数据目录约定(Market-1501 风格)
```
data_dir/
  bounding_box_train/  0001_c1s1_000151_00.jpg ...   # 前4位=身份ID
  bounding_box_test/   ...                            # gallery
  bounding_box_query/  ...                            # probe
```
> 注意：`train.py`/`datasets.py` 与 `eval.py` 必须读同一套目录命名。`eval.py` 读取的 query 目录名为 `bounding_box_query`、gallery 为 `bounding_box_test`，并兼容 `data_dir/pytorch/...` 子目录布局；若你的数据集 query 目录叫 `query/`，需重命名为 `bounding_box_query/` 或在 `eval.py` 的 `find_dir(...,'bounding_box_query')` 处对齐，否则会报 `FileNotFoundError`。

## 从头训练说明
- `--pretrained 1` 表示骨干用 ImageNet 预训练初始化(Re-ID 通用做法)。
- 基线与改进版**必须用相同的 --pretrained 值**，且都不要加载对方权重(不加 --resume/--load)。
- 唯一的变量就是 `--cbam_layers`，这样才能证明指标增益来自 CBAM。

## 评估接口

### 方式一：训练内置评估(evaluate.py)
训练脚本每 `--eval_start` 之后自动评估并保存 best_model.pth(按 mAP)。
也可在代码中调用 `evaluate.evaluate(model, query_loader, gallery_loader, device)`
得到 dict: `{rank1, rank5, rank10, mAP, num_query}`。
> 该方式会在每个评估 Epoch 把整批测试图读入内存，在 16G 内存机器上易触发 OOM，故训练长跑时常用 `--eval_start 1000` 跳过。

### 方式二：独立评估脚本 eval.py(推荐补指标用)
训练结束后单独加载权重补算指标，专为省内存设计：
- **流式提特征**：query/gallery 只通过 DataLoader 逐批读图，提完特征即释放，内存只保留 float32 特征矩阵(约几十 MB)；
- **分块算距离**：query×gallery 距离矩阵按 query 每 `--query_chunk` 条(默认 256)分块计算，不会再一次性生成 3368×19732 的巨型矩阵；
- **`--num_workers 2`、`pin_memory=False`**：规避子进程与锁页内存叠加导致的 OOM。

跑基线：
```
python eval.py --data_dir data/Market-1501-v15.09.15 \
    --weights runs/baseline/last_model.pth \
    --output_dir runs/baseline 2>&1 | tee eval_baseline.log
```
跑改进版(**CBAM 配置必须与训练时一致**)：
```
python eval.py --data_dir data/Market-1501-v15.09.15 \
    --weights runs/cbam_exp1/last_model.pth \
    --cbam_layers 2 3 4 --feat_dim 512 \
    --output_dir runs/cbam_exp1 2>&1 | tee eval_cbam.log
```

参数与输出说明：
- `--weights`：要评估的权重路径，如 `runs/baseline/last_model.pth`(兼容 `{'state_dict':...}` 与裸 state_dict)。
- **结构参数须与训练一致**：`--cbam_layers / --feat_dim / --per_block / --cbam_ratio / --cbam_kernel / --dropout` 任一与训练不符，`load_state_dict` 会因维度不匹配报错。
- `--num_classes` 默认 751(Market-1501 训练集身份数)，换数据集必须改，否则分类头维度对不上。
- 内存仍紧张时：把 `--num_workers 0`、`--query_chunk 64` 调小。
- 输出：终端打印 mAP / Rank-1 / Rank-5 / Rank-10 / Rank-20，并把关键指标写入 `<output_dir>/metrics.txt`，方便贴进中期报告。

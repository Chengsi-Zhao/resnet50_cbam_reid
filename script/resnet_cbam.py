# resnet_cbam.py
# ResNet-50 骨干网络，支持在各残差 stage 中按需嵌入 CBAM。
# per_block=True 时遵循原论文：CBAM 位于残差分支最后一个 BN 之后、shortcut 相加之前。
import os

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from cbam import CBAM


DEFAULT_MODEL_CACHE_DIR = '/home/myd/data/models'


def configure_model_cache(cache_dir=DEFAULT_MODEL_CACHE_DIR):
    """设置 torchvision 预训练权重缓存目录并返回实际 checkpoint 目录。"""
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    torch.hub.set_dir(cache_dir)
    return os.path.join(cache_dir, 'checkpoints')


class BottleneckWithCBAM(nn.Module):
    """把 CBAM 插入 torchvision Bottleneck 的残差分支末端。

    复用已经完成 ImageNet 初始化的卷积/BN，只改变 forward 的组合方式：
    conv3 -> bn3 -> CBAM -> add shortcut -> ReLU。
    """

    def __init__(self, block, channels, ratio=16, kernel_size=7,
                 attention_type='cbam'):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.conv3 = block.conv3
        self.bn3 = block.bn3
        self.relu = block.relu
        self.downsample = block.downsample
        self.stride = block.stride
        self.cbam = CBAM(
            channels, ratio=ratio, kernel_size=kernel_size,
            attention_type=attention_type
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class StageEndWithCBAM(nn.Module):
    """消融实验变体：整个 stage 完成后只使用一个 CBAM。"""

    def __init__(self, stage, channels, ratio=16, kernel_size=7,
                 attention_type='cbam'):
        super().__init__()
        self.blocks = stage
        self.cbam = CBAM(
            channels, ratio=ratio, kernel_size=kernel_size,
            attention_type=attention_type
        )

    def forward(self, x):
        return self.cbam(self.blocks(x))


class ResNet50CBAM(nn.Module):
    """
    Re-ID 特征提取骨干：ResNet-50 去掉原始分类头，替换为 embedding 层。
    参数：
      num_classes : 训练集身份数（ID loss 的分类类别数），由 train.py 注入。
      feat_dim    : 输出特征向量维度。
      cbam_layers : CBAM 嵌入哪些 stage，取值来自 {1,2,3,4}。
                    () = 纯基线；(2,3,4) = Re-ID 默认中深层配置。
      cbam_ratio  : 通道注意力降维比例。
      attention_type: channel / spatial / cbam，用于模块类型消融。
      per_block   : True=按原论文插入块内；False=该 stage 末尾嵌一次。
    训练 forward 返回 (feat, logits)；推理 extract_feature 返回 L2 归一化特征。
    """

    STAGE_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}

    def __init__(self, num_classes, feat_dim=512, cbam_layers=(2, 3, 4),
                 cbam_ratio=16, cbam_kernel=7, per_block=True,
                 pretrained=True, dropout=0.0,
                 model_cache_dir=DEFAULT_MODEL_CACHE_DIR,
                 attention_type='cbam'):
        super().__init__()
        if pretrained:
            checkpoint_dir = configure_model_cache(model_cache_dir)
            print(f'ImageNet 预训练权重缓存目录: {checkpoint_dir}')
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)

        invalid_layers = set(cbam_layers) - set(self.STAGE_CHANNELS)
        if invalid_layers:
            raise ValueError(f'cbam_layers 只能取 1/2/3/4，收到 {sorted(invalid_layers)}')

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = self._build_stage(
            base, 1, cbam_layers, cbam_ratio, cbam_kernel,
            per_block, attention_type
        )
        self.layer2 = self._build_stage(
            base, 2, cbam_layers, cbam_ratio, cbam_kernel,
            per_block, attention_type
        )
        self.layer3 = self._build_stage(
            base, 3, cbam_layers, cbam_ratio, cbam_kernel,
            per_block, attention_type
        )
        self.layer4 = self._build_stage(
            base, 4, cbam_layers, cbam_ratio, cbam_kernel,
            per_block, attention_type
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_dim = self.STAGE_CHANNELS[4]    # 2048
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.feat = nn.Linear(in_dim, feat_dim)
        # 标准 Re-ID BNNeck：度量损失用 BN 前特征，ID 分类用 BN 后特征。
        self.bn_feat = nn.BatchNorm1d(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)  # ID loss 分类头
        self.feat_dim = feat_dim
        nn.init.kaiming_normal_(self.feat.weight, mode='fan_out')
        nn.init.zeros_(self.feat.bias)
        nn.init.ones_(self.bn_feat.weight)
        nn.init.zeros_(self.bn_feat.bias)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)

    def _build_stage(self, base, idx, cbam_layers, ratio, kernel, per_block,
                     attention_type):
        stage = getattr(base, f'layer{idx}')
        if idx not in cbam_layers:
            return stage
        if not per_block:
            return StageEndWithCBAM(
                stage, self.STAGE_CHANNELS[idx], ratio, kernel, attention_type
            )
        return nn.Sequential(*[
            BottleneckWithCBAM(
                block, self.STAGE_CHANNELS[idx], ratio, kernel, attention_type
            )
            for block in stage
        ])

    def backbone_forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.backbone_forward(x)
        x = self.dropout(x)
        feat = self.feat(x)
        bn_feat = self.bn_feat(feat)
        logits = self.classifier(bn_feat)
        return feat, logits

    @torch.no_grad()
    def extract_feature(self, x):
        self.eval()
        x = self.backbone_forward(x)
        x = self.dropout(x)
        feat = self.feat(x)
        bn_feat = self.bn_feat(feat)
        return nn.functional.normalize(bn_feat, dim=1)


def build_model(cfg, num_classes):
    """按配置构建模型；cbam_layers=[] 即纯基线 ResNet-50。"""
    return ResNet50CBAM(
        num_classes=num_classes,
        feat_dim=cfg.feat_dim,
        cbam_layers=tuple(cfg.cbam_layers),
        cbam_ratio=cfg.cbam_ratio,
        cbam_kernel=cfg.cbam_kernel,
        per_block=cfg.per_block,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        model_cache_dir=getattr(
            cfg, 'model_cache_dir', DEFAULT_MODEL_CACHE_DIR
        ),
        attention_type=getattr(cfg, 'attention_type', 'cbam'),
    )

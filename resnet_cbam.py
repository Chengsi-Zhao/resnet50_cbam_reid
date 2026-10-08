# resnet_cbam.py
# ResNet-50 骨干网络，支持在 layer2/layer3/layer4 的残差块后按需嵌入 CBAM。
# cbam_layers 做成可配置参数，方便第三阶段消融实验（哪些层嵌、嵌几处）。
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from cbam import CBAM


class StageWithCBAM(nn.Module):
    """包装一个残差 stage：每个 bottleneck 块后追加一个 CBAM（per_block=True），
    或整个 stage 末尾只加一个 CBAM（per_block=False）。"""

    def __init__(self, stage, channels, ratio=16, kernel_size=7, per_block=True):
        super().__init__()
        self.blocks = stage
        self.per_block = per_block
        n = len(stage) if per_block else 1
        self.cbams = nn.ModuleList(
            [CBAM(channels, ratio=ratio, kernel_size=kernel_size) for _ in range(n)]
        )

    def forward(self, x):
        if self.per_block:
            for block, cbam in zip(self.blocks, self.cbams):
                x = block(x)
                x = cbam(x)
        else:
            for block in self.blocks:
                x = block(x)
            x = self.cbams[0](x)
        return x


class ResNet50CBAM(nn.Module):
    """
    Re-ID 特征提取骨干：ResNet-50 去掉原始分类头，替换为 embedding 层。
    参数：
      num_classes : 训练集身份数（ID loss 的分类类别数），由 train.py 注入。
      feat_dim    : 输出特征向量维度。
      cbam_layers : CBAM 嵌入哪些 stage，取值来自 {2,3,4}。
                    () = 纯基线 ResNet-50；(2,3,4) = 中深层全嵌。
      cbam_ratio  : 通道注意力降维比例。
      per_block   : True=每个残差块后都嵌；False=该 stage 末尾嵌一次。
    训练 forward 返回 (feat, logits)；推理 extract_feature 返回 L2 归一化特征。
    """

    STAGE_CHANNELS = {2: 512, 3: 1024, 4: 2048}

    def __init__(self, num_classes, feat_dim=512, cbam_layers=(2, 3, 4),
                 cbam_ratio=16, cbam_kernel=7, per_block=True,
                 pretrained=True, dropout=0.0):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)

        # 基线保持不变部分
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1          # layer1 通常不嵌 CBAM

        # 中深层按需嵌入 CBAM
        self.layer2 = self._build_stage(base, 2, cbam_layers, cbam_ratio,
                                        cbam_kernel, per_block)
        self.layer3 = self._build_stage(base, 3, cbam_layers, cbam_ratio,
                                        cbam_kernel, per_block)
        self.layer4 = self._build_stage(base, 4, cbam_layers, cbam_ratio,
                                        cbam_kernel, per_block)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_dim = self.STAGE_CHANNELS[4]    # 2048
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bn_feat = nn.BatchNorm1d(in_dim)
        self.feat = nn.Linear(in_dim, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)  # ID loss 分类头
        self.feat_dim = feat_dim

    def _build_stage(self, base, idx, cbam_layers, ratio, kernel, per_block):
        stage = getattr(base, f'layer{idx}')
        if idx in cbam_layers:
            return StageWithCBAM(stage, self.STAGE_CHANNELS[idx], ratio,
                                 kernel, per_block)
        return stage

    def backbone_forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.backbone_forward(x)
        x = self.dropout(x)
        x = self.bn_feat(x)
        feat = self.feat(x)
        logits = self.classifier(feat)
        return feat, logits

    @torch.no_grad()
    def extract_feature(self, x):
        self.eval()
        x = self.backbone_forward(x)
        x = self.bn_feat(x)
        feat = self.feat(x)
        return nn.functional.normalize(feat, dim=1)


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
    )

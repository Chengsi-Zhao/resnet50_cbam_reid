# cbam.py
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力: 对 H/W 空间维做全局平均/最大池化, 经共享 MLP 后相加, sigmoid 得到逐通道权重."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(channels // ratio, 1)
        # 用 Adaptive 池化模块做全局池化，输出恒为 1x1，避免手写 dim 的形状歧义
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        # 不做任何"置零"初始化：两个 1x1 卷积保持默认(Kaiming)初始化，
        # CBAM 一开始以 sigmoid(MLP输出)≈0.5 附近温和介入，能正常收到梯度。

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))   # [B,C,1,1]
        max_out = self.shared_mlp(self.max_pool(x))   # [B,C,1,1]
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力: 沿 channel 维做平均/最大, concat 后用大核卷积生成 HxW 权重图."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "spatial kernel_size only support 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)    # [B,1,H,W]
        max_out = x.amax(dim=1, keepdim=True)    # [B,1,H,W]
        att = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(att)


class CBAM(nn.Module):
    """通道注意力 -> 空间注意力 串联, 不改变张量形状, 可原位插入残差块之后."""

    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)   # 先通道
        x = x * self.spatial_att(x)   # 再空间
        return x
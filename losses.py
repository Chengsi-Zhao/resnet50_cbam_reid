# losses.py
# Re-ID 常用损失：身份交叉熵(ID loss) + batch-hard Triplet loss。
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Batch-hard triplet：对每个 anchor 取 batch 内最远的正样本、最近的负样本。"""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feat, labels):
        dist = torch.pairwise_distance(feat[:, None], feat[None, :])
        pos_mask = labels.eq(labels.view(-1, 1))
        pos_mask.fill_diagonal_(0)            # 排除自身
        neg_mask = ~labels.eq(labels.view(-1, 1))

        # hardest positive：同类中距离最大者
        pos_dist = dist.masked_fill(pos_mask == 0, 0).max(dim=1).values
        # hardest negative：异类中距离最小者
        neg_dist = dist.masked_fill(neg_mask == 0, float('inf')).min(dim=1).values

        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return feat.new_tensor(0.)
        loss = F.relu(pos_dist[valid] - neg_dist[valid] + self.margin)
        return loss.mean()


class ReidLoss(nn.Module):
    """组合损失 = id_weight * CE + triplet_weight * Triplet。"""

    def __init__(self, id_weight=1.0, triplet_weight=1.0, margin=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.triplet = TripletLoss(margin)
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight

    def forward(self, feat, logits, labels):
        loss_id = self.ce(logits, labels)
        loss_tri = self.triplet(F.normalize(feat, dim=1), labels)
        total = self.id_weight * loss_id + self.triplet_weight * loss_tri
        return total, loss_id.item(), loss_tri.item()

# evaluate.py
# 标准 Re-ID 评估：Rank-1/5/10 (CMC) 与 mAP。
# 逻辑同 Market-1501 官方 eval_reid：排除"同身份+同摄像头"的 distractor。
import re
import numpy as np
import torch


def _parse_pid_cam(path):
    name = path.split('/')[-1]
    m = re.match(r'(-?\d+)_c(\d+)', name)
    return int(m.group(1)), int(m.group(2))


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels, cams = [], [], []
    for imgs, lab, paths in loader:
        imgs = imgs.to(device)
        f = model.extract_feature(imgs)
        feats.append(f.cpu())
        for p in paths:
            pid, cam = _parse_pid_cam(p)
            labels.append(pid)
            cams.append(cam)
    feats = torch.cat(feats, 0).numpy()
    return feats, np.array(labels), np.array(cams)


def _distance(q, g):
    # 欧氏距离（特征已 L2 归一化）
    return np.linalg.norm(q[:, None, :] - g[None, :, :], axis=2)


def evaluate(model, query_loader, gallery_loader, device):
    qf, qp, qc = extract_features(model, query_loader, device)
    gf, gp, gc = extract_features(model, gallery_loader, device)
    distmat = _distance(qf, gf)

    cmc = np.zeros(distmat.shape[1])
    aps = []
    for i in range(distmat.shape[0]):
        order = np.argsort(distmat[i])
        matches = (gp[order] == qp[i])
        # 排除同摄像头的相同身份（distractor）
        keep = ~((gc[order] == qc[i]) & (gp[order] == qp[i]))
        valid = matches & keep
        if valid.sum() == 0:
            continue
        # CMC
        for r in range(distmat.shape[1]):
            cmc[r] += int(valid[:r + 1].any())
        # AP
        prec = np.cumsum(valid) / (np.arange(len(valid)) + 1)
        aps.append(prec[valid].mean())
    num_q = len(aps)
    cmc = cmc / max(num_q, 1) * 100
    mAP = np.mean(aps) * 100 if aps else 0.0
    return {'rank1': cmc[0], 'rank5': cmc[4] if len(cmc) > 4 else 0,
            'rank10': cmc[9] if len(cmc) > 9 else 0, 'mAP': mAP,
            'num_query': num_q}

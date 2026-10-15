# evaluate.py
# 标准 Re-ID 评估：Rank-1/5/10 (CMC) 与 mAP。
# 逻辑同 Market-1501 官方 eval_reid：保留 pid=0 distractor，
# 排除 pid=-1 junk 和"同身份+同摄像头"样本。
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


def _l2_normalize(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def compute_reid_metrics(qf, q_pids, q_cams, gf, g_pids, g_cams,
                         query_chunk=512):
    """按 Market-1501 协议分块计算指标，所有返回值统一为百分数。"""
    if len(q_pids) == 0 or len(g_pids) == 0:
        raise ValueError('query/gallery 不能为空')
    if query_chunk <= 0:
        raise ValueError('query_chunk 必须大于 0')

    qf = _l2_normalize(qf.astype(np.float32))
    gf = _l2_normalize(gf.astype(np.float32))
    ranks = np.zeros(4, dtype=np.float64)
    aps = []

    for start in range(0, len(q_pids), query_chunk):
        end = min(start + query_chunk, len(q_pids))
        order = np.argsort(-(qf[start:end] @ gf.T), axis=1, kind='mergesort')

        for offset, cur_order in enumerate(order):
            qi = start + offset
            # -1 和同身份同摄像头属于 junk，必须从排名中删除；
            # pid=0 distractor 作为负样本保留在排序中。
            remove = ((g_pids[cur_order] == -1) |
                      ((g_pids[cur_order] == q_pids[qi]) &
                       (g_cams[cur_order] == q_cams[qi])))
            cur_order = cur_order[~remove]
            matches = g_pids[cur_order] == q_pids[qi]
            hit = np.flatnonzero(matches)
            if len(hit) == 0:
                continue

            first = hit[0]
            for idx, rank in enumerate((1, 5, 10, 20)):
                if first < rank:
                    ranks[idx] += 1

            precision = np.cumsum(matches) / np.arange(1, len(matches) + 1)
            aps.append(float(precision[matches].mean()))

    num_valid = len(aps)
    scale = 100.0 / max(num_valid, 1)
    cmc = ranks * scale
    return {
        'mAP': float(np.mean(aps) * 100.0) if aps else 0.0,
        'rank1': float(cmc[0]),
        'rank5': float(cmc[1]),
        'rank10': float(cmc[2]),
        'rank20': float(cmc[3]),
        'num_query': num_valid,
    }


def evaluate(model, query_loader, gallery_loader, device, query_chunk=512):
    qf, qp, qc = extract_features(model, query_loader, device)
    gf, gp, gc = extract_features(model, gallery_loader, device)
    return compute_reid_metrics(qf, qp, qc, gf, gp, gc, query_chunk)

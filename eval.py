
# -*- coding: utf-8 -*-
"""
eval.py  --  Market-1501 ReID 独立评估脚本（省内存版）

设计要点：
1. 目录自动适配：query 依次尝试 bounding_box_query -> query，
   gallery 依次尝试 bounding_box_test -> gallery -> test；
   每个候选名先试 data_dir/pytorch/<名>，再试 data_dir/<名>，命中即返回。
2. 只留特征、不留图像：DataLoader 流式读图，提完即释放。
3. 距离矩阵分块计算：query 每 query_chunk 条一块，用归一化后的 q @ g.T 算相似度。
4. CMC 向量化，返回 rank1/5/10/20 与 mAP。
5. 打印指标并写入 output_dir/metrics.txt。

用法：
  基线：
    python eval.py --data_dir data/Market-1501-v15.09.15 \
        --weights runs/baseline/last_model.pth --output_dir runs/baseline
  改进版（CBAM 配置需与训练一致）：
    python eval.py --data_dir data/Market-1501-v15.09.15 \
        --weights runs/cbam_exp1/last_model.pth --cbam_layers 2 3 4 \
        --output_dir runs/cbam_exp1
"""
import os
import re
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from resnet_cbam import build_model


# ----------------------------- 命令行参数 -----------------------------
def parse_args():
    p = argparse.ArgumentParser('Market-1501 独立评估（省内存）')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='runs/eval')
    # 模型结构参数（必须与训练时保持一致）
    p.add_argument('--cbam_layers', type=int, nargs='+', default=[],
                   help='嵌入 CBAM 的 stage，如 2 3 4；基线留空')
    p.add_argument('--cbam_ratio', type=int, default=16)
    p.add_argument('--cbam_kernel', type=int, default=7)
    p.add_argument('--per_block', type=int, default=0)
    p.add_argument('--feat_dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--num_classes', type=int, default=751)
    p.add_argument('--pretrained', type=int, default=0,
                   help='评估会 load_state_dict 整体覆盖，无需拉预训练，默认 0')
    # 评估运行参数
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--query_chunk', type=int, default=512)
    p.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


# ----------------------------- 配置对象 -----------------------------
class _Cfg:
    pass


def build_cfg(args):
    cfg = _Cfg()
    for k in ('cbam_layers', 'cbam_ratio', 'cbam_kernel',
              'per_block', 'feat_dim', 'dropout', 'pretrained'):
        setattr(cfg, k, getattr(args, k))
    return cfg


# ----------------------------- 目录探测 -----------------------------
def find_dir(data_dir, names):
    """names 为候选名列表；每个候选先试 pytorch 子目录再试根目录。"""
    if isinstance(names, str):
        names = [names]
    for name in names:
        for cand in (os.path.join(data_dir, 'pytorch', name),
                     os.path.join(data_dir, name)):
            if os.path.isdir(cand):
                return cand
    raise FileNotFoundError('找不到目录 %s，请检查 data_dir 布局' % names)


# ----------------------------- 数据集 -----------------------------
IMG_EXT = ('.jpg', '.png', '.bmp', '.jpeg')


class ReIDDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.imgs = []      # (path, pid, camid)
        for fname in sorted(os.listdir(root)):
            if not fname.lower().endswith(IMG_EXT):
                continue
            pid, camid = self._parse(fname)
            if pid is None:
                continue
            self.imgs.append((os.path.join(root, fname), pid, camid))

    @staticmethod
    def _parse(fname):
        # Market-1501 命名：0001_c1s1_001051_03.jpg 或 0001_c1s1_001051.jpg
        m = re.match(r'^(\d+)_c(\d+)s?(\d*)', fname)
        if m:
            pid = int(m.group(1))
            camid = int(m.group(2)) - 1
        else:
            try:
                pid = int(fname.split('_')[0])
            except ValueError:
                return None, None
            camid = 0
        return pid, camid

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, pid, camid = self.imgs[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 256))
        return self.transform(img), pid, camid, path


def valid_index(imgs):
    """剔除 distractor：pid<=0 为干扰人。"""
    keep = [i for i, (_p, pid, _c) in enumerate(imgs) if pid > 0]
    return np.array(keep, dtype=np.int64)


# ----------------------------- 特征提取 -----------------------------
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, camids = [], [], []
    has_extract = hasattr(model, 'extract_feature')
    for imgs, pid, camid, _path in loader:
        imgs = imgs.to(device, non_blocking=False)
        if has_extract:
            f = model.extract_feature(imgs)
        else:
            out = model(imgs)
            f = out[0] if isinstance(out, (tuple, list)) else out
        feats.append(f.detach().float().cpu().numpy())
        pids.append(pid.numpy() if torch.is_tensor(pid) else np.array(pid))
        camids.append(camid.numpy() if torch.is_tensor(camid) else np.array(camid))
    feats = np.concatenate(feats, 0)
    pids = np.concatenate(pids, 0)
    camids = np.concatenate(camids, 0)
    return feats, pids, camids


def l2_norm(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ----------------------------- 指标计算 -----------------------------
def compute_metrics(qf, q_pids, q_cams, gf, g_pids, g_cams, chunk=512):
    n_q = len(q_pids)
    ranks = np.zeros(4, dtype=np.float64)  # rank1/5/10/20
    APs = []
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        sim = qf[s:e] @ gf.T          # 相似度，越大越相似
        order = np.argsort(-sim, axis=1, kind='mergesort')
        for i in range(e - s):
            qi = s + i
            cur_order = order[i]
            cur_pids = g_pids[cur_order]
            cur_cams = g_cams[cur_order]
            match = (cur_pids == q_pids[qi])
            same_cam = (cur_cams == q_cams[qi]) & (cur_pids == q_pids[qi])
            valid = ~same_cam
            if valid.sum() == 0:
                continue
            match_v = match[valid]
            hit = np.flatnonzero(match_v)
            if len(hit) == 0:
                continue
            first = hit[0]
            for k, kk in enumerate((1, 5, 10, 20)):
                if first < kk:
                    ranks[k] += 1
            cum_correct = np.cumsum(match_v)
            precision = cum_correct / np.arange(1, len(match_v) + 1)
            APs.append(precision[match_v].mean())
    n_valid = max(len(APs), 1)
    cmc = ranks / n_valid
    mAP = float(np.mean(APs)) if APs else 0.0
    return mAP, cmc


# ----------------------------- 主流程 -----------------------------
def run_eval(args, model, device):
    h, w = args.img_size
    tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    qdir = find_dir(args.data_dir, ['bounding_box_query', 'query'])
    gdir = find_dir(args.data_dir, ['bounding_box_test', 'gallery', 'test'])

    qset = ReIDDataset(qdir, tf)
    gset = ReIDDataset(gdir, tf)

    q_loader = DataLoader(qset, args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False)
    g_loader = DataLoader(gset, args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False)

    qf, q_pids, q_cams = extract_features(model, q_loader, device)
    gf, g_pids, g_cams = extract_features(model, g_loader, device)

    qk = valid_index(qset.imgs)
    gk = valid_index(gset.imgs)
    qf, q_pids, q_cams = qf[qk], q_pids[qk], q_cams[qk]
    gf, g_pids, g_cams = gf[gk], g_pids[gk], g_cams[gk]

    print('有效 query = %d，有效 gallery = %d' % (len(q_pids), len(g_pids)))

    qf = l2_norm(qf.astype(np.float32))
    gf = l2_norm(gf.astype(np.float32))

    mAP, cmc = compute_metrics(qf, q_pids, q_cams,
                               gf, g_pids, g_cams, chunk=args.query_chunk)
    return {
        'mAP': mAP,
        'rank1': float(cmc[0]),
        'rank5': float(cmc[1]),
        'rank10': float(cmc[2]),
        'rank20': float(cmc[3]),
    }


def main():
    args = parse_args()
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available()
                          and args.gpu >= 0 else 'cpu')

    cfg = build_cfg(args)
    model = build_model(cfg, args.num_classes).to(device)

    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    msg = model.load_state_dict(new_state, strict=False)
    print('加载权重 %s，CBAM=%s' % (args.weights, tuple(args.cbam_layers)))
    missing = getattr(msg, 'missing_keys', [])
    if missing:
        print('提示：未匹配的参数键 %d 个（结构参数与训练一致可忽略）' % len(missing))

    model.eval()
    m = run_eval(args, model, device)

    print('mAP    = %.4f' % m['mAP'])
    print('Rank-1 = %.4f' % m['rank1'])
    print('Rank-5 = %.4f' % m['rank5'])
    print('Rank-10 = %.4f' % m['rank10'])
    print('Rank-20 = %.4f' % m['rank20'])

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'metrics.txt')
    with open(out, 'w', encoding='utf-8') as f:
        for k in ('mAP', 'rank1', 'rank5', 'rank10', 'rank20'):
            f.write('%s = %.4f\n' % (k, m[k]))
    print('指标已写入 %s' % out)


if __name__ == '__main__':
    main()
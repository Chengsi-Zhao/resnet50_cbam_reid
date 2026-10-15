
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
    python script/eval.py --data_dir data/Market-1501-v15.09.15 \
        --weights model/baseline/last_model.pth --output_dir result/baseline/eval
  改进版（新版 checkpoint 会自动恢复模型配置）：
    python script/eval.py --data_dir data/Market-1501-v15.09.15 \
        --weights model/cbam_exp1/last_model.pth \
        --output_dir result/cbam_exp1/eval
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
from evaluate import compute_reid_metrics


# ----------------------------- 命令行参数 -----------------------------
def parse_args():
    p = argparse.ArgumentParser('Market-1501 独立评估（省内存）')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='result/eval')
    # 模型结构参数（必须与训练时保持一致）
    p.add_argument('--cbam_layers', type=int, nargs='*', default=None,
                   choices=(1, 2, 3, 4),
                   help='默认从 checkpoint 读取；传空参数表示纯基线')
    p.add_argument('--cbam_ratio', type=int, default=None)
    p.add_argument('--cbam_kernel', type=int, choices=(3, 7), default=None)
    p.add_argument('--attention_type', type=str,
                   choices=('channel', 'spatial', 'cbam'), default=None)
    p.add_argument('--per_block', type=int, choices=(0, 1), default=None)
    p.add_argument('--feat_dim', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--num_classes', type=int, default=None)
    # 评估运行参数
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--query_chunk', type=int, default=512)
    p.add_argument('--img_size', type=int, nargs=2, default=None,
                   help='默认从 checkpoint 读取，否则使用 256 128')
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


# ----------------------------- 配置对象 -----------------------------
class _Cfg:
    pass


DEFAULT_MODEL_CONFIG = {
    'cbam_layers': [2, 3, 4],
    'cbam_ratio': 16,
    'cbam_kernel': 7,
    'attention_type': 'cbam',
    'per_block': 1,
    'feat_dim': 512,
    'dropout': 0.0,
}


def resolve_model_config(args, checkpoint):
    saved = checkpoint.get('model_config', {}) if isinstance(checkpoint, dict) else {}
    for key, default in DEFAULT_MODEL_CONFIG.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, default))
    if args.num_classes is None:
        args.num_classes = (checkpoint.get('num_classes', 751)
                            if isinstance(checkpoint, dict) else 751)
    if args.img_size is None:
        args.img_size = (checkpoint.get('image_size', [256, 128])
                         if isinstance(checkpoint, dict) else [256, 128])
    args.per_block = bool(args.per_block)


def build_cfg(args):
    cfg = _Cfg()
    for key in DEFAULT_MODEL_CONFIG:
        setattr(cfg, key, getattr(args, key))
    cfg.pretrained = False
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
        # 数据损坏应显式报错，不能用空白图悄悄污染评估指标。
        img = Image.open(path).convert('RGB')
        return self.transform(img), pid, camid, path


def valid_index(imgs, keep_distractors=False):
    """query 仅保留正 ID；gallery 保留 pid=0 distractor，剔除 pid=-1 junk。"""
    keep = [
        i for i, (_p, pid, _c) in enumerate(imgs)
        if (pid >= 0 if keep_distractors else pid > 0)
    ]
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


# 指标实现统一复用 evaluate.compute_reid_metrics。


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
    if len(qset) == 0 or len(gset) == 0:
        raise ValueError('query/gallery 目录中没有可评估图片')

    q_loader = DataLoader(qset, args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False)
    g_loader = DataLoader(gset, args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False)

    qf, q_pids, q_cams = extract_features(model, q_loader, device)
    gf, g_pids, g_cams = extract_features(model, g_loader, device)

    qk = valid_index(qset.imgs)
    gk = valid_index(gset.imgs, keep_distractors=True)
    qf, q_pids, q_cams = qf[qk], q_pids[qk], q_cams[qk]
    gf, g_pids, g_cams = gf[gk], g_pids[gk], g_cams[gk]

    print('有效 query = %d，有效 gallery = %d' % (len(q_pids), len(g_pids)))

    return compute_reid_metrics(
        qf, q_pids, q_cams, gf, g_pids, g_cams, args.query_chunk
    )


def main():
    args = parse_args()
    if args.query_chunk <= 0:
        raise ValueError('--query_chunk 必须大于 0')
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available()
                          and args.gpu >= 0 else 'cpu')

    checkpoint = torch.load(args.weights, map_location='cpu')
    resolve_model_config(args, checkpoint)
    cfg = build_cfg(args)
    model = build_model(cfg, args.num_classes).to(device)

    state = (checkpoint['state_dict']
             if isinstance(checkpoint, dict) and 'state_dict' in checkpoint
             else checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    # 严格加载：模型结构不一致时立即报错，禁止静默漏载 CBAM 参数。
    model.load_state_dict(state, strict=True)
    print('加载权重 %s，CBAM=%s，per_block=%s' %
          (args.weights, tuple(args.cbam_layers), args.per_block))

    model.eval()
    m = run_eval(args, model, device)

    print('mAP     = %.2f%%' % m['mAP'])
    print('Rank-1  = %.2f%%' % m['rank1'])
    print('Rank-5  = %.2f%%' % m['rank5'])
    print('Rank-10 = %.2f%%' % m['rank10'])
    print('Rank-20 = %.2f%%' % m['rank20'])

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'metrics.txt')
    with open(out, 'w', encoding='utf-8') as f:
        for k in ('mAP', 'rank1', 'rank5', 'rank10', 'rank20'):
            f.write('%s_percent = %.4f\n' % (k, m[k]))
    print('指标已写入 %s' % out)


if __name__ == '__main__':
    main()

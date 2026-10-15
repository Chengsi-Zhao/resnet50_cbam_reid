# train.py
# ResNet-50(+CBAM) Re-ID 训练入口，改进版与基线超参逐项对齐。
# 用法示例（改进版，中深层全嵌 CBAM）：
#   python script/train.py --data_dir /data/market1501 --cbam_layers 2 3 4 \
#       --epochs 60 --batch_size 32 --lr 3.5e-4 --output_dir model/cbam_exp1
# 基线（不加 CBAM）：把 --cbam_layers 置空（--cbam_layers 后不填数字）。
import os
import argparse
import time
import random
import numpy as np
import torch
from torch.optim import Adam

from resnet_cbam import build_model
from datasets import build_train, build_query_gallery
from losses import ReidLoss
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--cbam_layers', type=int, nargs='*', default=[2, 3, 4],
                   choices=(1, 2, 3, 4),
                   help='CBAM 嵌入的 stage，取值 1/2/3/4；置空即基线')
    p.add_argument('--cbam_ratio', type=int, default=16)
    p.add_argument('--cbam_kernel', type=int, choices=(3, 7), default=7)
    p.add_argument('--attention_type', type=str,
                   choices=('channel', 'spatial', 'cbam'), default='cbam',
                   help='注意力类型：仅通道、仅空间或完整 CBAM')
    p.add_argument('--per_block', type=int, choices=(0, 1), default=1)
    p.add_argument('--feat_dim', type=int, default=512)
    p.add_argument('--pretrained', type=int, choices=(0, 1), default=1,
                   help='ImageNet 预训练骨干；基线与改进版必须设成相同值')
    p.add_argument('--model_cache_dir', type=str,
                   default='/home/myd/data/models',
                   help='torchvision 预训练权重缓存根目录')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=128)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--pid_batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3.5e-4)
    p.add_argument('--backbone_lr_mult', type=float, default=1.0,
                   help='预训练主干相对基础学习率的倍率')
    p.add_argument('--cbam_lr_mult', type=float, default=1.0,
                   help='注意力模块相对基础学习率的倍率')
    p.add_argument('--head_lr_mult', type=float, default=1.0,
                   help='Re-ID 特征头和分类头相对基础学习率的倍率')
    p.add_argument('--warmup_epochs', type=int, default=0)
    p.add_argument('--warmup_start_factor', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--lr_step', type=int, default=40)
    p.add_argument('--id_weight', type=float, default=1.0)
    p.add_argument('--triplet_weight', type=float, default=1.0)
    p.add_argument('--margin', type=float, default=0.3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--eval_start', type=int, default=10)
    p.add_argument('--output_dir', type=str, default='model/exp')
    p.add_argument('--seed', type=int, default=1)
    return p.parse_args()


def make_checkpoint(model, args, num_classes, epoch, metrics=None):
    config_keys = ('cbam_layers', 'cbam_ratio', 'cbam_kernel', 'per_block',
                   'attention_type', 'feat_dim', 'dropout')
    config = {key: getattr(args, key) for key in config_keys}
    config['cbam_layers'] = list(config['cbam_layers'])
    training_keys = (
        'lr', 'backbone_lr_mult', 'cbam_lr_mult', 'head_lr_mult',
        'warmup_epochs', 'warmup_start_factor', 'weight_decay', 'lr_step',
        'batch_size', 'pid_batch_size', 'id_weight', 'triplet_weight',
        'margin', 'seed',
    )
    return {
        'state_dict': model.state_dict(),
        'model_config': config,
        'training_config': {
            key: getattr(args, key) for key in training_keys
        },
        'num_classes': num_classes,
        'image_size': [args.img_h, args.img_w],
        'epoch': epoch,
        'metrics': metrics or {},
    }


def build_optimizer(model, args):
    """把预训练主干、CBAM 和 Re-ID 头分组，支持独立学习率。"""
    grouped = {'backbone': [], 'cbam': [], 'head': []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.cbam.' in name:
            grouped['cbam'].append(param)
        elif name.startswith(('feat.', 'bn_feat.', 'classifier.')):
            grouped['head'].append(param)
        else:
            grouped['backbone'].append(param)

    multipliers = {
        'backbone': args.backbone_lr_mult,
        'cbam': args.cbam_lr_mult,
        'head': args.head_lr_mult,
    }
    param_groups = []
    for name, params in grouped.items():
        if not params:
            continue
        base_lr = args.lr * multipliers[name]
        param_groups.append({
            'params': params,
            'lr': base_lr,
            'base_lr': base_lr,
            'group_name': name,
        })
    return Adam(param_groups, weight_decay=args.weight_decay)


def set_epoch_lrs(optimizer, epoch, args):
    """先线性 warmup，再按 lr_step 衰减；epoch 从 1 开始。"""
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        if args.warmup_epochs == 1:
            scale = 1.0
        else:
            progress = (epoch - 1) / (args.warmup_epochs - 1)
            scale = (
                args.warmup_start_factor
                + (1.0 - args.warmup_start_factor) * progress
            )
    else:
        completed = max(epoch - args.warmup_epochs - 1, 0)
        scale = 0.1 ** (completed // args.lr_step)

    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * scale


def main():
    args = parse_args()
    if args.lr <= 0 or args.lr_step <= 0:
        raise ValueError('--lr 和 --lr_step 必须大于 0')
    if args.warmup_epochs < 0:
        raise ValueError('--warmup_epochs 不能小于 0')
    if not 0 < args.warmup_start_factor <= 1:
        raise ValueError('--warmup_start_factor 必须在 (0, 1] 内')
    for key in ('backbone_lr_mult', 'cbam_lr_mult', 'head_lr_mult'):
        if getattr(args, key) <= 0:
            raise ValueError(f'--{key} 必须大于 0')
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size = (args.img_h, args.img_w)
    train_loader, num_classes, iters = build_train(
        args.data_dir, img_size, args.batch_size,
        args.pid_batch_size, args.num_workers)
    q_loader, g_loader = build_query_gallery(
        args.data_dir, img_size, args.batch_size, args.num_workers)

    args.per_block = bool(args.per_block)
    model = build_model(args, num_classes).to(device)
    print(
        f'CBAM 嵌入位置 cbam_layers={tuple(args.cbam_layers)}，'
        f'attention_type={args.attention_type}，身份数={num_classes}'
    )
    total = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total:,}')

    criterion = ReidLoss(args.id_weight, args.triplet_weight, args.margin)
    optimizer = build_optimizer(model, args)
    group_sizes = {
        group['group_name']: sum(p.numel() for p in group['params'])
        for group in optimizer.param_groups
    }
    print(f'优化器参数分组: {group_sizes}')

    best_map = -1
    data_iter = iter(train_loader)
    for epoch in range(1, args.epochs + 1):
        set_epoch_lrs(optimizer, epoch, args)
        model.train()
        t0 = time.time()
        running = 0.0
        for step in range(iters):
            imgs, labels, _ = next(data_iter)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            feat, logits = model(imgs)
            loss, _, _ = criterion(feat, logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        lr_text = ','.join(
            f"{group['group_name']}={group['lr']:.2e}"
            for group in optimizer.param_groups
        )
        msg = (f'[Epoch {epoch:02d}/{args.epochs}] '
               f'loss={running / max(iters, 1):.4f} '
               f'lr[{lr_text}] '
               f'time={time.time() - t0:.1f}s')
        if epoch >= args.eval_start:
            metrics = evaluate(model, q_loader, g_loader, device)
            msg += (f" | mAP={metrics['mAP']:.2f} "
                    f"Rank1={metrics['rank1']:.2f} "
                    f"Rank5={metrics['rank5']:.2f}")
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                torch.save(
                    make_checkpoint(model, args, num_classes, epoch, metrics),
                    os.path.join(args.output_dir, 'best_model.pth'),
                )
        print(msg, flush=True)

    torch.save(
        make_checkpoint(model, args, num_classes, args.epochs),
        os.path.join(args.output_dir, 'last_model.pth'),
    )
    best_text = f'{best_map:.2f}%' if best_map >= 0 else '未评估'
    print(f'训练结束，best mAP={best_text}，权重保存于 {args.output_dir}')


if __name__ == '__main__':
    main()

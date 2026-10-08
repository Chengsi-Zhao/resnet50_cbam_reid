# train.py
# ResNet-50(+CBAM) Re-ID 训练入口。从头训练，改进版与基线超参逐项对齐。
# 用法示例（改进版，中深层全嵌 CBAM）：
#   python train.py --data_dir /data/market1501 --cbam_layers 2 3 4 \
#       --epochs 60 --batch_size 32 --lr 3.5e-4 --output_dir runs/cbam_exp1
# 基线（不加 CBAM）：把 --cbam_layers 置空（--cbam_layers 后不填数字）。
import os
import argparse
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from resnet_cbam import build_model
from datasets import build_train, build_query_gallery
from losses import ReidLoss
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--cbam_layers', type=int, nargs='*', default=[2, 3, 4],
                   help='CBAM 嵌入的 stage，取值 2/3/4；置空即基线 ResNet-50')
    p.add_argument('--cbam_ratio', type=int, default=16)
    p.add_argument('--cbam_kernel', type=int, default=7)
    p.add_argument('--per_block', type=int, default=1)
    p.add_argument('--feat_dim', type=int, default=512)
    p.add_argument('--pretrained', type=int, default=1,
                   help='ImageNet 预训练骨干；基线与改进版必须设成相同值')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=128)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--pid_batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3.5e-4)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--lr_step', type=int, default=40)
    p.add_argument('--id_weight', type=float, default=1.0)
    p.add_argument('--triplet_weight', type=float, default=1.0)
    p.add_argument('--margin', type=float, default=0.3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--eval_start', type=int, default=10)
    p.add_argument('--output_dir', type=str, default='runs/exp')
    p.add_argument('--seed', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size = (args.img_h, args.img_w)
    train_loader, num_classes, iters = build_train(
        args.data_dir, img_size, args.batch_size,
        args.pid_batch_size, args.num_workers)
    q_loader, g_loader = build_query_gallery(
        args.data_dir, img_size, args.batch_size, args.num_workers)

    args.per_block = bool(args.per_block)
    model = build_model(args, num_classes).to(device)
    print(f'CBAM 嵌入位置 cbam_layers={tuple(args.cbam_layers)}，身份数={num_classes}')
    total = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total:,}')

    criterion = ReidLoss(args.id_weight, args.triplet_weight, args.margin)
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    best_map = -1
    data_iter = iter(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for step in range(iters):
            imgs, labels, _ = next(data_iter)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            feat, logits = model(imgs)
            loss, li, lt = criterion(feat, logits, labels)
            loss.backward()
            optimizer.step()
            running += li + lt
        scheduler.step()

        msg = (f'[Epoch {epoch:02d}/{args.epochs}] '
               f'loss_id+tri={running / max(iters, 1):.4f} '
               f'lr={scheduler.get_last_lr()[0]:.2e} '
               f'time={time.time() - t0:.1f}s')
        if epoch >= args.eval_start:
            metrics = evaluate(model, q_loader, g_loader, device)
            msg += (f" | mAP={metrics['mAP']:.2f} "
                    f"Rank1={metrics['rank1']:.2f} "
                    f"Rank5={metrics['rank5']:.2f}")
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best_model.pth'))
        print(msg, flush=True)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'last_model.pth'))
    print(f'训练结束，best mAP={best_map:.2f}，权重保存于 {args.output_dir}')


if __name__ == '__main__':
    main()

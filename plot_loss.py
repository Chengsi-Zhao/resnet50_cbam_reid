# plot_loss.py
import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 兼容 loss_id+tri / loss_id / loss 等常见写法
PATTERN = re.compile(
    r'\[Epoch\s+(\d+)\s*/\s*(\d+)\].*?loss[_\w+]*\s*=\s*([\d.eE+-]+)'
)

def parse_log(path):
    epochs, losses = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(3)))
    # 同一 epoch 可能被重复记录，保留最后一次
    merged = dict(zip(epochs, losses))
    ep = sorted(merged)
    return ep, [merged[e] for e in ep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='+', required=True,
                    help='日志文件列表，如 baseline.log runs/cbam_noeval/train.log')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='与 --logs 一一对应的图例标签，缺省则用文件名')
    ap.add_argument('--out', default='loss_compare.png')
    ap.add_argument('--title', default='Training Loss Comparison')
    ap.add_argument('--log_scale', action='store_true', help='y 轴对数刻度')
    ap.add_argument('--smooth', type=int, default=0,
                    help='滑动平均窗口，0 表示不平滑')
    args = ap.parse_args()

    labels = args.labels or [l.split('/')[-1].replace('.log', '') for l in args.logs]
    if len(labels) != len(args.logs):
        raise ValueError('--labels 数量必须与 --logs 一致')

    fig, ax = plt.subplots(figsize=(9, 5))
    for path, label in zip(args.logs, labels):
        epochs, losses = parse_log(path)
        if not epochs:
            print(f'[警告] {path} 未解析到 loss，已跳过')
            continue
        print(f'{label}: {len(epochs)} 个 epoch，'
              f'最小 loss = {min(losses):.4f} @ epoch {epochs[losses.index(min(losses))]}')

        if args.smooth > 1:
            w = args.smooth
            sm = [sum(losses[max(0, i-w+1):i+1]) / len(losses[max(0, i-w+1):i+1])
                  for i in range(len(losses))]
            ax.plot(epochs, sm, linewidth=1.6, label=f'{label} (smooth w={w})')
            ax.plot(epochs, losses, alpha=0.3, linewidth=0.8)
        else:
            ax.plot(epochs, losses, marker='o', markersize=3,
                    linewidth=1.2, label=label)

    if args.log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'已保存: {args.out}')

if __name__ == '__main__':
    main()
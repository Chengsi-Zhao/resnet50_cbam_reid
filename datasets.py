# datasets.py
# 加载 Market-1501 风格的行人 Re-ID 数据集。
# 目录约定：
#   <root>/bounding_box_train/0001_c1s1_000151_00.jpg   (前4位=身份ID)
#   <root>/bounding_box_test/...
#   <root>/query/...
# 文件名第一个数字为身份 ID；'-1' 开头为 distractor，自动忽略。
import os
import re
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image


def _parse_pid(filename):
    m = re.match(r'(-?\d+)_', filename)
    return int(m.group(1)) if m else None


def _collect(dir_path):
    items = []
    for name in sorted(os.listdir(dir_path)):
        if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        pid = _parse_pid(name)
        if pid is None or pid <= 0:
            continue
        items.append((os.path.join(dir_path, name), pid))
    return items


def build_transform(img_size=(256, 128), is_train=True):
    """训练/测试 transform。改进版与基线必须完全一致，逐项对齐。"""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Pad(10),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        norm,
    ])


class ReidDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path


class IdRandomSampler(Sampler):
    """ID-based 采样：每个 batch 取 pbs 个身份、每个身份 ipp 张，
    满足 triplet loss 对'同 batch 含同 ID 多张'的要求。"""

    def __init__(self, pid2idx, pbs, ipp):
        self.pid2idx = pid2idx
        self.pbs = pbs
        self.ipp = ipp
        self.pids = list(pid2idx.keys())

    def __iter__(self):
        while True:  # 无限流，配合 train.py 里的 iterations/epoch 控制步数
            random.shuffle(self.pids)
            for start in range(0, len(self.pids), self.pbs):
                for p in self.pids[start:start + self.pbs]:
                    idxs = self.pid2idx[p]
                    for i in random.choices(idxs, k=self.ipp):
                        yield i

    def __len__(self):
        return len(self.pids) * self.ipp


def build_train(root, img_size, batch_size, pid_batch_size=8, num_workers=4):
    train_items = _collect(os.path.join(root, 'bounding_box_train'))
    pids = sorted(set(p for _, p in train_items))
    pid2label = {p: i for i, p in enumerate(pids)}
    relabeled = [(path, pid2label[p]) for path, p in train_items]
    ds = ReidDataset(relabeled, build_transform(img_size, True))

    pid2idx = defaultdict(list)
    for i, (_, label) in enumerate(relabeled):
        pid2idx[label].append(i)
    ipp = max(1, batch_size // pid_batch_size)
    sampler = IdRandomSampler(pid2idx, pid_batch_size, ipp)
    loader = DataLoader(ds, batch_size=pid_batch_size * ipp, sampler=sampler,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    # iterations/epoch 近似为原样本数 / batch_size
    iters_per_epoch = max(1, len(relabeled) // (pid_batch_size * ipp))
    return loader, len(pids), iters_per_epoch


def build_query_gallery(root, img_size, batch_size, num_workers=4):
    t = build_transform(img_size, False)
    query = ReidDataset(_collect(os.path.join(root, 'query')), t)
    gallery = ReidDataset(_collect(os.path.join(root, 'bounding_box_test')), t)
    q_loader = DataLoader(query, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    g_loader = DataLoader(gallery, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return q_loader, g_loader

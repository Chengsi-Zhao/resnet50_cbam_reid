import os
from PIL import Image

root = 'data/fake_market'
splits = {
    'bounding_box_train': 16,   # 训练身份数，要 >= pid_batch_size(默认8)，给16稳妥
    'bounding_box_test': 16,    # gallery 底库
    'query': 16,                # 查询集
}
for split, n_pid in splits.items():
    d = os.path.join(root, split)
    os.makedirs(d, exist_ok=True)
    for pid in range(1, n_pid + 1):
        cnt = 4 if split == 'bounding_box_train' else 1  # 训练每身份4张，评测1张
        for idx in range(cnt):
            img = Image.new('RGB', (128, 256), (pid * 10 % 255, idx * 30, 80))
            # query 与 gallery 必须来自不同摄像头，否则会被协议视为 junk。
            cam = 2 if split == 'bounding_box_test' else 1
            name = f'{pid:04d}_c{cam}s1_{pid:03d}0{idx}_0{idx}.jpg'
            img.save(os.path.join(d, name))
print('fake data generated under', root)
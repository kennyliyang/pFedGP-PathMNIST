import pickle
import numpy as np
import os
from medmnist import PathMNIST

print("正在下载并加载 PathMNIST...")
train_dataset = PathMNIST(split="train", download=True)
test_dataset = PathMNIST(split="test", download=True)

# 尺寸补齐 (28x28 -> 32x32)
def pad_to_32(imgs):
    return np.pad(imgs, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

print("正在重新打包数据 (严格匹配 NoisyClients 格式)...")
data_dict = {
    'train': {
        'data': pad_to_32(train_dataset.imgs),
        # 核心修复点：明确命名为 'label'，并转换为 PyTorch 喜欢的整数类型
        'label': train_dataset.labels.squeeze().astype(np.int64) 
    },
    'test': {
        'data': pad_to_32(test_dataset.imgs),
        'label': test_dataset.labels.squeeze().astype(np.int64)
    }
}

# 确保文件精准存入 datafolder
save_path = os.path.expanduser('~/pFedGP/experiments/datafolder/medmnist_path_dictionary.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"🎉 最终版数据包已就绪！保存至 {save_path}")
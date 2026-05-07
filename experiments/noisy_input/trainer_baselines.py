import sys
import os
# 确保能找到根目录的 pFedGP 和 utils 模块
sys.path.append(os.path.abspath('../../'))

import argparse
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import copy

from experiments.backbone import CNNTarget
from experiments.noisy_input.clients import NoisyClients
from utils import get_device, set_logger, set_seed, save_experiment, calc_metrics

parser = argparse.ArgumentParser(description="Baselines: FedAvg & FedPer")

parser.add_argument("--data-path", type=str, default='../datafolder/medmnist_path_dictionary.pkl')
parser.add_argument("--data-name", type=str, default="pathmnist", choices=['cifar10', 'cifar100', 'pathmnist'])
parser.add_argument("--method", type=str, default="fedavg", choices=['fedavg', 'fedper'])
parser.add_argument("--num-classes", type=int, default=None, help="number of classes")

# 训练参数
parser.add_argument("--num-steps", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--inner-steps", type=int, default=1)
parser.add_argument("--num-client-agg", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=1e-3)
parser.add_argument("--n-kernels", type=int, default=16)
parser.add_argument('--embed-dim', type=int, default=84)
parser.add_argument("--gpus", type=str, default='0')
parser.add_argument("--eval-every", type=int, default=25)
parser.add_argument("--save-path", type=str, default="../output/Baselines")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
set_logger()
set_seed(args.seed)
device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

DEFAULT_CLASSES = {'cifar10': 10, 'cifar100': 100, 'pathmnist': 9}

if args.num_classes is not None:
    num_classes = args.num_classes
else:
    num_classes = DEFAULT_CLASSES.get(args.data_name)
    if num_classes is None:
        raise ValueError(f" 未知的数据集 '{args.data_name}'，请明确传入 --num-classes 参数！")

exp_name = f'Baseline_{args.method}_data_{args.data_name}_seed_{args.seed}'
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)

# 1. 加载数据
clients = NoisyClients(data_path=args.data_path, batch_size=args.batch_size)
num_clients = len(clients)

# 2. 初始化全局特征提取器 (Base)
global_base = CNNTarget(n_kernels=args.n_kernels, embedding_dim=args.embed_dim).to(device)

# 3. 初始化分类器 (Head)
if args.method == 'fedavg':
    # FedAvg: 全局只有一个共享的大脑
    global_head = nn.Linear(args.embed_dim, num_classes).to(device)
else:
    # FedPer: 每个客户端私有一个大脑
    local_heads = {i: nn.Linear(args.embed_dim, num_classes).to(device) for i in range(num_clients)}

criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def eval_model(split):
    results = defaultdict(lambda: defaultdict(list))
    
    for client_id in range(num_clients):
        running_loss, running_correct, running_samples = 0., 0., 0.
        curr_data = clients.test_loaders[client_id] if split == 'test' else clients.val_loaders[client_id]
        
        # 组装模型
        curr_base = copy.deepcopy(global_base).eval()
        curr_head = copy.deepcopy(global_head).eval() if args.method == 'fedavg' else local_heads[client_id].eval()

        for batch in curr_data:
            img, label = batch[0].to(device), batch[1].to(device)
            features = curr_base(img)
            logits = curr_head(features)
            loss = criterion(logits, label)
            
            running_loss += loss.item()
            running_correct += logits.argmax(1).eq(label).sum().item()
            running_samples += len(label)
            
        results[client_id]['loss'] = running_loss / len(curr_data)
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples
        
    return results

# 训练循环
best_acc = -1
step_iter = trange(args.num_steps)

for step in step_iter:
    client_ids = np.random.choice(range(num_clients), size=args.num_client_agg, replace=False)
    
    # 存放准备平均的参数
    base_params = OrderedDict((n, torch.zeros_like(p)) for n, p in global_base.named_parameters())
    if args.method == 'fedavg':
        head_params = OrderedDict((n, torch.zeros_like(p)) for n, p in global_head.named_parameters())
        
    for client_id in client_ids:
        curr_base = copy.deepcopy(global_base).train()
        curr_head = copy.deepcopy(global_head).train() if args.method == 'fedavg' else local_heads[client_id].train()
        
        optimizer = torch.optim.SGD(list(curr_base.parameters()) + list(curr_head.parameters()), lr=args.lr, weight_decay=args.wd)
        
        # 本地训练
        for _ in range(args.inner_steps):
            for batch in clients.train_loaders[client_id]:
                img, label = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                logits = curr_head(curr_base(img))
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                
        # 累加身体参数
        for n, p in curr_base.named_parameters():
            base_params[n] += p.data
            
        # 累加或保留大脑参数
        if args.method == 'fedavg':
            for n, p in curr_head.named_parameters():
                head_params[n] += p.data
        else:
            local_heads[client_id] = copy.deepcopy(curr_head) # FedPer绝对不上传，直接本地覆盖

    # Server 端平均并更新全局模型
    for n, p in base_params.items():
        base_params[n] = p / args.num_client_agg
    global_base.load_state_dict(base_params)
    
    if args.method == 'fedavg':
        for n, p in head_params.items():
            head_params[n] = p / args.num_client_agg
        global_head.load_state_dict(head_params)

    # 评估与日志打印
    if (step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps:
        val_results = eval_model(split="val")
        val_avg_loss, val_avg_acc = calc_metrics(val_results)
        logging.info(f"Step: {step + 1}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_avg_acc:.4f}")
        if val_avg_acc > best_acc:
            best_acc = val_avg_acc
            
test_results = eval_model(split="test")
test_avg_loss, test_avg_acc = calc_metrics(test_results)
logging.info(f"\n🚀 Baseline 完成！ Final Test Loss: {test_avg_loss:.4f}, Test Acc: {test_avg_acc:.4f}")
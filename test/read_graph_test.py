import torch
from collections import Counter
import numpy as np

# def get_label(sample):
#     user_graph = sample
#     label = int(user_graph["user"].y.item())  # 转为 int
#     return label

# def count_labels(dataset):
#     counter = Counter(get_label(sample) for sample in dataset)
#     return dict(counter)

# 加载图（从 pt 文件）
all_graphs = torch.load("data\graph\merged_sample_20250816_164358.pt")
# train_graphs = torch.load("data/graph/train_samples.pt")
# evaluate_graphs = torch.load("data/graph/test_samples.pt")

# print(f"总样本数: {len(all_graphs)}")
# print("训练集类别分布:", count_labels(train_graphs))
# print("测试集类别分布:", count_labels(evaluate_graphs))


# for data in all_graphs:
#     print(data['user'].x)

# print(data)
# print(data['user'])
# print(data['terminal'])
# print(data['vm'])
# print(data['user', 'connects', 'terminal'])
# print(data['terminal', 'connects', 'vm'])

# X, y = [], []
# for g in all_graphs:
#     ux = g["user"].x
#     ux = ux.squeeze(0).cpu().numpy()
#     uy = g["user"].y
#     uy = uy.item()
#     X.append(ux)
#     y.append(uy)
# print(np.asarray(X).shape, np.asarray(y).shape)
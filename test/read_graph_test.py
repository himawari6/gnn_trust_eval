import torch

# 加载图（从 pt 文件）
all_graphs = torch.load("data/graph/sample_1.pt")

for data in all_graphs:
    print(data)
    print(data['user'].y)
# print(data)
# print(data['user'])
# print(data['terminal'])
# print(data['vm'])
# print(data['user', 'connects', 'terminal'])
# print(data['terminal', 'connects', 'vm'])
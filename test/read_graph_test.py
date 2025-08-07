import torch

# 加载图（从 pt 文件）
all_graphs = torch.load("data/graph/sample_1_old.pt")
data = all_graphs[7]  # 取第一个用户图
print(data)
print(data['user'])
print(data['terminal'])
print(data['vm'])
print(data['user', 'connects', 'terminal'])
print(data['terminal', 'connects', 'vm'])
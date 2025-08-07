import torch
from model.model import HeteroTrustGNN

# 加载图（从 pt 文件）
all_graphs = torch.load("data/graph/sample_1.pt")
data = all_graphs[7]  # 取第一个用户图

model = HeteroTrustGNN()
model.eval()

with torch.no_grad():
    logits = model(data)  # shape: [1, num_classes]
    probs = torch.softmax(logits, dim=-1)
    print("Predicted probability distribution:", probs)

import torch
from model.model import HeteroTrustGNN
from model.ablation_model import HeteroTrustGraphProp

# 加载 .pt 文件 (List[HeteroData])
# data = torch.load("data/graph/toy.pt")
data = torch.load("data/graph/toy_alert_on_node.pt")

# model = HeteroTrustGNN()
# model.eval()

# with torch.no_grad():
#     logits = model(data)  # (N_user, num_classes) 常见 N_user=1
#     probs = torch.softmax(logits, dim=-1)
#     print("logits:", logits)
#     print("probs:", probs)

model = HeteroTrustGraphProp()
model.eval()

with torch.no_grad():
    logits = model(data)  # (N_user, num_classes) 常见 N_user=1
    probs = torch.softmax(logits, dim=-1)
    print("logits:", logits)
    print("probs:", probs)


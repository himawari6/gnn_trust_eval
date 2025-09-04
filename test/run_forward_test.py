import torch
from model.model import HeteroTrustGNN

# 加载 .pt 文件 (List[HeteroData])
data = torch.load("data/graph/toy.pt")

model = HeteroTrustGNN(
    vm_in_dim=7, term_in_dim=2, user_in_dim=8,
    edge_dim=2, user_hidden_dim=32, terminal_hidden_dim=8, vm_hidden_dim=64, num_layers=3, num_classes=3
)
model.eval()

with torch.no_grad():
    logits = model(data)  # (N_user, num_classes) 常见 N_user=1
    probs = torch.softmax(logits, dim=-1)
    print("logits:", logits)
    print("probs:", probs)


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from model.model import HeteroTrustGNN  # 你的模型
import os
from datetime import datetime

# ---------------------------
# 加载数据
# ---------------------------
def load_graph_dataset(pt_files):
    dataset = []
    for file in pt_files:
        data_list = torch.load(file)  # List[HeteroData]
        dataset.extend(data_list)
    return dataset

train_files = ["train_sample_01.pt", "train_sample_02.pt"]  # 训练数据
train_dataset = load_graph_dataset(train_files)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# ---------------------------
# 初始化模型
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroTrustGNN(
    vm_in_dim=7,
    term_in_dim=2,
    user_in_dim=8,
    edge_dim=2,
    user_hidden_dim=32,
    terminal_hidden_dim=8,
    vm_hidden_dim=32,
    num_layers=2,
    num_classes=3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 训练
# ---------------------------
def train_model(num_epochs=20):
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            loss = criterion(out, data["user"].y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 保存模型
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"result/train/model/trust_gnn_model{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {os.path.abspath(model_path)}")

    # 绘制 loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"loss_curve_of_model_at_{current_time}.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    train_model(num_epochs=1)

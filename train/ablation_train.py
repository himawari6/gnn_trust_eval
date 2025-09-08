import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from model.ablation_model import HeteroTrustGraphProp  # 你的模型
import os
from datetime import datetime
from utils.logger import get_logger

# ---------------------------
# 加载数据
# ---------------------------
def load_graph_dataset(pt_files):
    dataset = []
    for file in pt_files:
        data_list = torch.load(file)  # List[HeteroData]
        dataset.extend(data_list)
    return dataset

train_files = ["data/graph/train_samples_alert_on_node.pt"]  # 训练数据
train_dataset = load_graph_dataset(train_files)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

logger = get_logger(log_dir="log", log_name="GNN_train")

# ---------------------------
# 初始化模型
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroTrustGraphProp(
    user_hidden_dim=32,
    terminal_hidden_dim=8,
    vm_hidden_dim=32
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
scheduler = ExponentialLR(optimizer, gamma=0.95)

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
            
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Loss: {avg_loss:.4f}, "
            f"LR={scheduler.get_last_lr()[0]:.6f}"
        )

    # 保存模型
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"result/train/model/trust_basegraph_model{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到 {os.path.abspath(model_path)}")

    # 绘制 loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"result/train/figures/loss_curve_of_basegraph_model_at_{current_time}.png")
    plt.show()

if __name__ == "__main__":
    train_model(num_epochs=100)

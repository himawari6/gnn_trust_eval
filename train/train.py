import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

from model.hgnn_model import HeteroTrustGNN
from utils.logger import get_logger
from config.ablation_config import ABLATION_CONFIGS

# ---------------------------
# 数据加载
# ---------------------------
def load_graph_dataset(pt_files):
    dataset = []
    for file in pt_files:
        data_list = torch.load(file)
        dataset.extend(data_list)
    return dataset

# ---------------------------
# 训练函数
# ---------------------------
def train_model(model, train_loader, optimizer, scheduler, criterion,
                device, logger, num_epochs, save_tag):

    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

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
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # ---------------------------
    # 保存模型 & 曲线
    # ---------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = (
        f"result/train/model/"
        f"{save_tag}_gnn_model_{timestamp}.pth"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到 {os.path.abspath(model_path)}")

    fig_path = (
        f"result/train/figures/"
        f"loss_curve_{save_tag}_{timestamp}.png"
    )
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss ({save_tag})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"Loss 曲线已保存到 {os.path.abspath(fig_path)}")


# ---------------------------
# 主入口
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train HeteroTrustGNN")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=ABLATION_CONFIGS.keys(),
        help="Ablation mode"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = ABLATION_CONFIGS[args.mode]

    # ---------------------------
    # Logger
    # ---------------------------
    logger = get_logger(
        log_dir="log",
        log_name=f"GNN_train_{cfg.tag}"
    )
    
    # 这里的tag是自然语言的，如FullGraph，对应mode的UTV
    logger.info(f"开始训练，消融模式：{cfg.tag}")

    # ---------------------------
    # 数据
    # ---------------------------
    if cfg.use_vm2user_edge:
        train_files = ["data/graph/train/train_samples_with_vm2user_edge.pt"]
    else:
        train_files = ["data/graph/train/train_samples.pt"]

    train_dataset = load_graph_dataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # ---------------------------
    # 模型
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeteroTrustGNN(
        vm_in_dim=7,
        term_in_dim=2,
        user_in_dim=8,
        edge_dim=2,
        user_hidden_dim=32,
        terminal_hidden_dim=8,
        vm_hidden_dim=32,
        num_layers=2,
        num_classes=3,
        mode=cfg.mode
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # ---------------------------
    # 训练
    # ---------------------------
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        logger=logger,
        num_epochs=args.epochs,
        save_tag=cfg.tag
    )


if __name__ == "__main__":
    main()

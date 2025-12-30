import os
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from model.gcn_model import HomoTrustGNN_GCN
from model.graphsage_model import HomoTrustGNN_GraphSAGE

from utils.logger import get_logger

# ---------------------------
# 数据加载
# ---------------------------
def load_graph_dataset(pt_files):
    dataset = []
    for file in pt_files:
        data_list = torch.load(file)
        dataset.extend(data_list)
    return dataset

def build_homognn_model(model_type: str, node_dim: int):
    if model_type == 'gcn':
        return HomoTrustGNN_GCN(in_dim=node_dim)
    elif model_type == 'sage':
        return HomoTrustGNN_GraphSAGE(in_dim=node_dim)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")
    
def train_homognn_model(model, train_loader, optimizer, scheduler, criterion,
                        device, logger, num_epochs, model_tag):
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            # 清除上一组（批，batch）data留下的梯度
            optimizer.zero_grad()

            # 输出, forward
            out = model(data)
            # 比对输出和标签
            loss = criterion(out, data.y[data.user_mask])
            
            # 计算梯度
            loss.backward()
            # 根据梯度，由优化器改变参数
            optimizer.step()

            # 累计这一组数据的loss
            total_loss += loss.item()

        scheduler.step()

        # 这时得到了所有组的损失，除以组数得到每一组平均损失
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = (
        f"result/train/model/"
        f"{model_tag}_homognn_model_{timestamp}.pth"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到 {os.path.abspath(model_path)}")

    fig_path = (
        f"result/train/figures/"
        f"loss_curve_homognn_{model_tag}_{timestamp}.png"
    )
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss of ({model_tag})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"Loss 曲线已保存到 {os.path.abspath(fig_path)}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['gcn', 'sage'],
        help="choose_homo_model"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model_type = args.model

    logger = get_logger(
        log_dir="log",
        log_name=f"homo_GNN_train_{model_type}"
    )
    logger.info(f"开始训练，同构图模型：{model_type}")

    train_files = ['data/graph/train/train_samples_homo.pt']
    train_dataset = load_graph_dataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 取第二个维度的size，例如对矩阵就输出列数，也就是一行里面的元素数。
    # 对一个样本而言，因为x是m行n列（m个节点，每个结点的特征数为n，所以size(1)取的是特征数
    model = build_homognn_model(
        model_type=model_type, 
        node_dim=train_dataset[0].x.size(1)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    train_homognn_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        logger=logger,
        num_epochs=args.epochs,
        model_tag=model_type
    )

if __name__ == '__main__':
    main()


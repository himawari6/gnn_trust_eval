import os
import argparse

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

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
# 评估函数
# ---------------------------
def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)

            preds = out.argmax(dim=1).cpu().numpy()
            labels = data["user"].y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_labels, all_preds


# ---------------------------
# 主入口
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate HeteroTrustGNN")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=ABLATION_CONFIGS.keys()
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.pth)"
    )
    args = parser.parse_args()

    cfg = ABLATION_CONFIGS[args.mode]

    logger = get_logger(
        log_dir="log",
        log_name=f"GNN_evaluate_{cfg.tag}"
    )

    logger.info(f"开始评估，消融模式：{cfg.tag}")
    logger.info(f"模型路径：{args.model_path}")

    # ---------------------------
    # 数据
    # ---------------------------
    if cfg.use_vm2user_edge:
        test_files = ["data/graph/evaluate/evaluate_samples_with_vm2user_edge.pt"]
    else:
        test_files = ["data/graph/evaluate/evaluate_samples.pt"]

    test_dataset = load_graph_dataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在：{args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # ---------------------------
    # 评估
    # ---------------------------
    labels, preds = evaluate(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    logger.info(
        f"Accuracy: {acc:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    logger.info(
        "\n" + classification_report(
            labels, preds, digits=4, zero_division=0
        )
    )


if __name__ == "__main__":
    main()

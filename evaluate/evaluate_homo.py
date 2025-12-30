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
    
def evaluate_homognn_model(model, dataloader, device):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)

            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y[data.user_mask].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_labels, all_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['gcn', 'sage'],
        help="choose_homo_model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.pth)"
    )
    args = parser.parse_args()

    model_type = args.model

    logger = get_logger(
        log_dir="log",
        log_name=f"homo_GNN_evaluate_{model_type}"
    )

    logger.info(f"开始评估，同构图模型：{model_type}")
    logger.info(f"模型路径：{args.model_path}")

    test_files = ['data/graph/evaluate/evaluate_samples_homo.pt']
    test_dataset = load_graph_dataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_homognn_model(
        model_type=model_type, 
        node_dim=test_dataset[0].x.size(1)
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在：{args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # ---------------------------
    # 评估
    # ---------------------------
    labels, preds = evaluate_homognn_model(model, test_loader, device)

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


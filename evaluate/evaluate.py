import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from model.model import HeteroTrustGNN  # 你的模型
import os
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

test_files = ["data/graph/evaluate_samples.pt"]  # 测试数据
test_dataset = load_graph_dataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

logger = get_logger(log_dir="log", log_name="GNN_evaluate")

# ---------------------------
# 初始化模型并加载权重
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

model_path = "result\\train\\model\\trust_gnn_model20250818_115927.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"找不到模型文件 {model_path}，请先运行 train.py")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------------------
# 测试
# ---------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data["user"].y.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

logger.info(
    f"Accuracy: {acc:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall: {recall:.4f}\n"
    f"F1-score: {f1:.4f}\n"
)
logger.info("Classification Report:\n", classification_report(all_labels, all_preds, zero_division=0))

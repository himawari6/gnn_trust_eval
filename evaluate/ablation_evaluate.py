import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from model.ablation_model import HeteroTrustGraphProp  # 你的模型
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

test_files = ["data/graph/evaluate_samples_alert_on_node.pt"]  # 测试数据
test_dataset = load_graph_dataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

logger = get_logger(log_dir="log", log_name="BaseGraph_evaluate")

# ---------------------------
# 初始化模型并加载权重
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroTrustGraphProp(
    user_hidden_dim=32,
    terminal_hidden_dim=8,
    vm_hidden_dim=32
).to(device)

model_path = "result\\train\\model\\trust_basegraph_model20250908_215244.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"找不到模型文件 {model_path}，请先运行 ablation_train.py")

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

logger.info(f'模型：{model_path}')
logger.info(
    f"Accuracy: {acc:.4f}, "
    f"Precision: {precision:.4f}, "
    f"Recall: {recall:.4f}, "
    f"F1-score: {f1:.4f}"
)
logger.info('\n' + classification_report(all_labels, all_preds, digits=4, zero_division=0))

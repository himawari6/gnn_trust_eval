import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import VMToTerminalLayer, TerminalToUserLayer

class HeteroTrustGNN(nn.Module):
    def __init__(self, 
                 vm_in_dim=7, term_in_dim=2, user_in_dim=8, 
                 edge_dim=2, hidden_dim=32, num_classes=3):
        super().__init__()

        self.vm_to_term = VMToTerminalLayer(vm_in_dim, edge_dim, term_in_dim, hidden_dim)
        self.term_to_user = TerminalToUserLayer(hidden_dim, edge_dim, user_in_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        vm_x = data["vm"].x
        term_x = data["terminal"].x
        user_x = data["user"].x

        # VM -> Terminal 聚合
        term_x_updated = self.vm_to_term(
            vm_x, term_x,
            data["terminal", "connects", "vm"].edge_index,
            data["terminal", "connects", "vm"].edge_attr
        )

        # Terminal -> User 聚合
        user_x_updated = self.term_to_user(
            term_x_updated, user_x,
            data["user", "connects", "terminal"].edge_index,
            data["user", "connects", "terminal"].edge_attr
        )

        # 分类器输出
        out = self.classifier(user_x_updated)  # shape: [1, num_classes]
        return out

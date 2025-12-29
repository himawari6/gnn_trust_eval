import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

class HomoTrustGNN_GraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_classes: int = 3,
        # dropout: float = 0.3,
    ):
        super().__init__()

        self.num_layers = num_layers
        # self.dropout = dropout

        # ---------- GNN layers ----------
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # ---------- classifier ----------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        """
        data: torch_geometric.data.Data
        必须包含:
          - x
          - edge_index
          - user_mask
        """
        x, edge_index = data.x, data.edge_index

        # ----- GNN propagation -----
        for conv in self.convs:
            x = conv(x, edge_index)
            # x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- only classify user nodes -----
        user_x = x[data.user_mask]          # 通过mask索引，最后user_x里面只有是user的节点
        out = self.classifier(user_x)

        return out

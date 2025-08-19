import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model.modules import VMToTerminalLayer, TerminalToUserLayer

class HeteroTrustGNN(nn.Module):
    def __init__(self,
                 vm_in_dim=7, term_in_dim=2, user_in_dim=8,
                 edge_dim=2, user_hidden_dim=32, terminal_hidden_dim=8, vm_hidden_dim=32, 
                 num_layers=2, num_classes=3):
        """
        hidden_dim: 投影到的维度
        num_layers: 重复的层数（每层包含 VM->T 和 T->U）
        """
        super().__init__()
        self.user_hidden_dim = user_hidden_dim
        self.terminal_hidden_dim = terminal_hidden_dim
        self.vm_hidden_dim = vm_hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 投影层：把原始特征投影到 hidden_dim
        self.vm_proj = nn.Linear(vm_in_dim, vm_hidden_dim)
        self.term_proj = nn.Linear(term_in_dim, terminal_hidden_dim)
        self.user_proj = nn.Linear(user_in_dim, user_hidden_dim)

        # 构造重复的层
        self.vm_to_term_layers = nn.ModuleList([
            VMToTerminalLayer(vm_hidden_dim, terminal_hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.term_to_user_layers = nn.ModuleList([
            TerminalToUserLayer(terminal_hidden_dim, user_hidden_dim, edge_dim) for _ in range(num_layers)
        ])

        # 最后的分类器（把 user_hidden -> logits）
        self.classifier = nn.Sequential(
            nn.Linear(user_hidden_dim, user_hidden_dim),
            nn.ReLU(),
            nn.Linear(user_hidden_dim, num_classes)
        )

    def forward(self, data):
        """
        data: HeteroData for one graph or batched graphs
        Expect node features:
          data['vm'].x  (N_vm, vm_in_dim)
          data['terminal'].x (N_term, term_in_dim)
          data['user'].x (N_user, user_in_dim)
        Expect edges:
          data['terminal','connects','vm'].edge_index  shape [2,E1] (term_idx, vm_idx)
          data['terminal','connects','vm'].edge_attr   shape [E1, edge_dim]
          data['user','connects','terminal'].edge_index shape [2, E2] (user_idx, term_idx)
          data['user','connects','terminal'].edge_attr  shape [E2, edge_dim]
        """
        # If node types missing, fallback: just classify user features
        if ('terminal' not in data.node_types) or ('vm' not in data.node_types):
            user_x = self.user_proj(data['user'].x)     # (N_user, H)
            return self.classifier(user_x)

        # initial projection
        vm_x = self.vm_proj(data['vm'].x)           # (N_vm, H)
        term_x = self.term_proj(data['terminal'].x) # (N_term, H)
        user_x = self.user_proj(data['user'].x)     # (N_user, H)

        # edge features
        edge_tv_node_index = data['terminal', 'connects', 'vm'].edge_index
        edge_ut_node_index = data['user', 'connects', 'terminal'].edge_index
        edge_tv_attr = data['terminal', 'connects', 'vm'].edge_attr  # (E1, edge_dim)
        edge_ut_attr = data['user', 'connects', 'terminal'].edge_attr  # (E2, edge_dim)

        # 逐层聚合
        for l in range(self.num_layers):
            # VM -> Terminal
            agg_term = self.vm_to_term_layers[l](
                vm_x, term_x,
                edge_tv_node_index,
                edge_tv_attr
            )  # (N_term, H)

            # 残差更新 terminal 节点
            term_x = term_x + agg_term

            # Terminal -> User
            agg_user = self.term_to_user_layers[l](
                term_x, user_x,
                edge_ut_node_index,
                edge_ut_attr
            )  # (N_user, H)

            # 残差更新 user 节点
            user_x = user_x + agg_user


        # 最终分类（基于 user_x）
        # graph_embedding = global_mean_pool(user_x, data["user"].batch)
        # out = self.classifier(graph_embedding)
        out = self.classifier(user_x)
        return out

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum

class VMToTerminalLayer(nn.Module):
    """
    一个消息生成器：VM -> Terminal
    输入：vm_x (N_vm, H), term_x (N_term, H), edge_index (2, E) 其中 edge_index[0]=term_idx, edge_index[1]=vm_idx
          edge_attr (E, edge_dim)
    输出：agg_messages -> (N_term, H)
    """
    def __init__(self, vm_hidden_dim, terminal_hidden_dim, edge_dim):
        super().__init__()
        # 输入维度： vm(H) + edge(edge_dim) + term(H)
        self.mlp = nn.Sequential(
            nn.Linear(vm_hidden_dim + edge_dim + terminal_hidden_dim, vm_hidden_dim),
            nn.ReLU(),
            nn.Linear(vm_hidden_dim, terminal_hidden_dim),
            nn.ReLU()
        )

    def forward(self, vm_x, term_x, edge_index, edge_attr):
        # edge_index: [2, E], 约定为 [term_idx, vm_idx]
        term_idx = edge_index[0]
        vm_idx = edge_index[1]

        vm_feat = vm_x[vm_idx]          # (E, H)
        term_feat = term_x[term_idx]    # (E, H)
        edge_feat = edge_attr           # (E, edge_dim)

        message_input = torch.cat([vm_feat, edge_feat, term_feat], dim=1)  # (E, 2H+edge_dim)
        messages = self.mlp(message_input)  # (E, H)

        # 聚合到 term 节点（可能有多个消息到同一个 term）
        agg = scatter_sum(messages, term_idx, dim=0, dim_size=term_x.size(0))  # (N_term, H)
        return agg


class TerminalToUserLayer(nn.Module):
    """
    Terminal -> User 聚合层
    输入：term_x (N_term, H), user_x (N_user, H), edge_index (2, E2) 约定为 [user_idx, term_idx]
    输出：agg_messages -> (N_user, H)
    """
    def __init__(self, terminal_hidden_dim, user_hidden_dim, edge_dim):
        super().__init__()
        # 输入维度： term(H) + edge(edge_dim) + user(H)
        self.mlp = nn.Sequential(
            nn.Linear(terminal_hidden_dim + edge_dim + user_hidden_dim, user_hidden_dim),
            nn.ReLU(),
            nn.Linear(user_hidden_dim, user_hidden_dim),
            nn.ReLU()
        )

    def forward(self, term_x, user_x, edge_index, edge_attr):
        # edge_index: [2, E], 约定为 [user_idx, term_idx]
        user_idx = edge_index[0]
        term_idx = edge_index[1]

        term_feat = term_x[term_idx]    # (E, H)
        user_feat = user_x[user_idx]    # (E, H)
        edge_feat = edge_attr           # (E, edge_dim)

        message_input = torch.cat([term_feat, edge_feat, user_feat], dim=1)  # (E, 2H+edge_dim)
        messages = self.mlp(message_input)  # (E, H)

        # 聚合到 user 节点（通常每个图 user 数量小，常为1）
        agg = scatter_sum(messages, user_idx, dim=0, dim_size=user_x.size(0))  # (N_user, H)
        return agg


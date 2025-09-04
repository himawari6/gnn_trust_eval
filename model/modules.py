import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class VMToTerminalLayer(MessagePassing):
    def __init__(self, vm_hidden_dim, terminal_hidden_dim, edge_dim):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(vm_hidden_dim + edge_dim + terminal_hidden_dim, terminal_hidden_dim),
            nn.ReLU(),
            nn.Linear(terminal_hidden_dim, terminal_hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        src_x, dst_x = x
        return self.propagate(edge_index=edge_index, x=(src_x, dst_x), edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        # x_j: vm 的特征, x_i: terminal 的特征
        msg_input = torch.cat([x_j, edge_attr, x_i], dim=-1)
        return self.mlp(msg_input)

    def update(self, aggr_out, x):
        # 残差更新 terminal
        return x + aggr_out


class TerminalToUserLayer(MessagePassing):
    def __init__(self, terminal_hidden_dim, user_hidden_dim, edge_dim):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(terminal_hidden_dim + edge_dim + user_hidden_dim, user_hidden_dim),
            nn.ReLU(),
            nn.Linear(user_hidden_dim, user_hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        src_x, dst_x = x
        return self.propagate(edge_index=edge_index, x=(src_x, dst_x), edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        # x_j: terminal 的特征, x_i: user 的特征
        msg_input = torch.cat([x_j, edge_attr, x_i], dim=-1)
        return self.mlp(msg_input)

    def update(self, aggr_out, x):
        return x + aggr_out




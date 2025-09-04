import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class VMToTerminalLayer(MessagePassing):
    def __init__(self, vm_hidden_dim, terminal_hidden_dim, edge_dim):
        super().__init__(aggr='add', flow="target_to_source")
        self.mlp = nn.Sequential(
            nn.Linear(vm_hidden_dim + edge_dim + terminal_hidden_dim, vm_hidden_dim),
            nn.ReLU(),
            nn.Linear(vm_hidden_dim, terminal_hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edges):
        return self.propagate(
            edge_index=edges['edge_index'],
            vm_x=x[0],
            term_x=x[1],
            edge_attr=edges['edge_attr']
        )

    def message(self, vm_x_j, term_x_i, edge_attr):
        msg_input = torch.cat([vm_x_j, edge_attr, term_x_i], dim=-1)
        return self.mlp(msg_input)

    def update(self, aggr_out, term_x):
        # 残差更新
        return term_x + aggr_out


class TerminalToUserLayer(MessagePassing):
    def __init__(self, terminal_hidden_dim, user_hidden_dim, edge_dim):
        super().__init__(aggr='add', flow="target_to_source")
        self.mlp = nn.Sequential(
            nn.Linear(terminal_hidden_dim + edge_dim + user_hidden_dim, user_hidden_dim),
            nn.ReLU(),
            nn.Linear(user_hidden_dim, user_hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edges):
        return self.propagate(
            edge_index=edges['edge_index'],
            term_x=x[0],
            user_x=x[1],
            edge_attr=edges['edge_attr']
        )

    def message(self, term_x_j, user_x_i, edge_attr):
        msg_input = torch.cat([term_x_j, edge_attr, user_x_i], dim=-1)
        return self.mlp(msg_input)

    def update(self, aggr_out, user_x):
        # 残差更新
        return user_x + aggr_out



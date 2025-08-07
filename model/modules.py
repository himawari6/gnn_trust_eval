import torch
import torch.nn as nn
import torch_scatter

class VMToTerminalLayer(nn.Module):
    def __init__(self, vm_dim, edge_dim, terminal_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vm_dim + edge_dim + terminal_dim, hidden_dim),
            nn.ReLU()
        )

    # 输入：
    def forward(self, vm_x, term_x, edge_index, edge_attr):
        src, dst = edge_index  # src: Terminal idx, dst: VM idx

        vm_feat = vm_x[dst]
        term_feat = term_x[src]
        edge_feat = edge_attr

        message_input = torch.cat([vm_feat, edge_feat, term_feat], dim=1)
        messages = self.mlp(message_input)

        # 聚合多个 VM → 同一个 Terminal
        agg_messages = torch_scatter.scatter_mean(messages, src, dim=0, dim_size=term_x.size(0))
        return agg_messages

class TerminalToUserLayer(nn.Module):
    def __init__(self, term_dim, edge_dim, user_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(term_dim + edge_dim + user_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, term_x, user_x, edge_index, edge_attr):
        src, dst = edge_index  # src: User idx (always 0), dst: Terminal idx

        term_feat = term_x[dst]
        user_feat = user_x[src]
        edge_feat = edge_attr

        message_input = torch.cat([term_feat, edge_feat, user_feat], dim=1)
        messages = self.mlp(message_input)

        # 所有 Terminal → 聚合成一个用户嵌入（只有一个用户）
        agg_messages = torch_scatter.scatter_mean(messages, src, dim=0, dim_size=user_x.size(0))
        return agg_messages

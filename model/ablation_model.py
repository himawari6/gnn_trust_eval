import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import HeteroConv
from torch_geometric.data import HeteroData
from torch_scatter import scatter


class VMToTerminalLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # 聚合方式是 sum，加权后就是加权和

    def forward(self, x, edge_index, edge_attr):
        normed_login_time = torch.log1p(edge_attr) # 得到的是个张量，应该是所有边的log(1+连接时间)
        vm_idx, terminal_idx = edge_index
        denominators = scatter(normed_login_time, terminal_idx, reduce='sum')
        weight = normed_login_time / (denominators[terminal_idx] + 1e-8)
        return self.propagate(edge_index=edge_index, x=x, edge_weight=weight)
    
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
    
    def update(self, aggr_out, x):
        vm_scores, terminal_scores = x
        return terminal_scores + aggr_out
    
class TerminalToUserLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # 聚合方式是 sum，加权后就是加权和

    def forward(self, x, edge_index, edge_attr):
        normed_login_time = torch.log1p(edge_attr) # 得到的是个张量，应该是所有边的log(1+连接时间)
        terminal_idx, user_idx = edge_index
        denominators = scatter(normed_login_time, user_idx, reduce='sum')
        weight = normed_login_time / (denominators[user_idx] + 1e-8)
        return self.propagate(edge_index=edge_index, x=x, edge_weight=weight)
    
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
    
    def update(self, aggr_out, x):
        terminal_scores, user_score = x
        return user_score + aggr_out
    

class HeteroTrustGraphProp(nn.Module):
    def __init__(self,
                 vm_in_dim=8, term_in_dim=3, user_in_dim=8,
                 edge_dim=1, user_hidden_dim=8, terminal_hidden_dim=8, vm_hidden_dim=8,
                 num_layers=1, num_classes=3):
        super().__init__()

        self.vm_trust_calculator = nn.Sequential(
            nn.Linear(vm_in_dim, vm_hidden_dim),
            nn.ReLU(),
            nn.Linear(vm_hidden_dim, 1)
        )
        self.terminal_trust_calculator = nn.Sequential(
            nn.Linear(term_in_dim, terminal_hidden_dim),
            nn.ReLU(),
            nn.Linear(terminal_hidden_dim, 1)
        )
        self.user_trust_calculator = nn.Sequential(
            nn.Linear(user_in_dim, user_hidden_dim),
            nn.ReLU(),
            nn.Linear(user_hidden_dim, 1)
        )

        self.conv = HeteroConv({
            ('vm', 'accessed_by', 'terminal'): VMToTerminalLayer(),
            ('terminal', 'used_by', 'user'): TerminalToUserLayer(),
        })

        self.classifier = nn.Linear(1, num_classes)

    def forward(self, data: HeteroData):
        if ('terminal' not in data.node_types) or ('vm' not in data.node_types):
            user_trust_score = self.user_trust_calculator(data['user'].x)
            return self.classifier(user_trust_score)
        
        initial_trust_scores = {
            'vm': self.vm_trust_calculator(data['vm'].x),
            'terminal': self.terminal_trust_calculator(data['terminal'].x),
            'user': self.user_trust_calculator(data['user'].x),
        }

        edge_index = data.edge_index_dict
        edge_attr = data.edge_attr_dict

        user_trust_score = self.conv(initial_trust_scores, edge_index, edge_attr)['user']

        out = self.classifier(user_trust_score)
        return out
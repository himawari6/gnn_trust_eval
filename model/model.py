import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv
from torch_geometric.data import HeteroData
from model.modules import VMToTerminalLayer, TerminalToUserLayer, VMToUserLayer


class HeteroTrustGNN(nn.Module):
    def __init__(self,
                 vm_in_dim=7, term_in_dim=2, user_in_dim=8,
                 edge_dim=2, user_hidden_dim=32, terminal_hidden_dim=8, vm_hidden_dim=32,
                 num_layers=2, num_classes=3, mode='UTV'):
        super().__init__()

        assert mode in ['U', 'UT', 'UV', 'UTV']
        self.mode = mode
        self.num_layers = num_layers

        # 投影层
        self.vm_proj = nn.Sequential(
            nn.Linear(vm_in_dim, vm_hidden_dim),
        )
        self.term_proj = nn.Sequential(
            nn.Linear(term_in_dim, terminal_hidden_dim),
        )
        self.user_proj = nn.Sequential(
            nn.Linear(user_in_dim, user_hidden_dim),
        )

        # 构造多层 HeteroConv
        # self.layers = nn.ModuleList()
        # for _ in range(num_layers):
        #     conv = HeteroConv({
        #         ('vm', 'accessed_by', 'terminal'): VMToTerminalLayer(vm_hidden_dim, terminal_hidden_dim, edge_dim),
        #         ('terminal', 'used_by', 'user'): TerminalToUserLayer(terminal_hidden_dim, user_hidden_dim, edge_dim),
        #     })
        #     self.layers.append(conv)
        # ---------- HeteroConv layers ----------
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            relations = {}

            # VM -> Terminal（只在完整模型中使用）
            if mode in ['UTV']:
                relations[('vm', 'accessed_by', 'terminal')] = \
                    VMToTerminalLayer(
                        vm_hidden_dim,
                        terminal_hidden_dim,
                        edge_dim
                    )
            
            # Terminal -> User
            if mode in ['UT', 'UTV']:
                relations[('terminal', 'used_by', 'user')] = \
                    TerminalToUserLayer(
                        terminal_hidden_dim,
                        user_hidden_dim,
                        edge_dim
                    )

            # VM -> User
            if mode in ['UV']:
                relations[('vm', 'employed_by', 'user')] = \
                    VMToUserLayer(
                        vm_hidden_dim,
                        user_hidden_dim,
                        edge_dim
                    )

            self.layers.append(HeteroConv(relations))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(user_hidden_dim, user_hidden_dim),
            nn.ReLU(),
            nn.Linear(user_hidden_dim, num_classes)
        )

    def forward(self, data: HeteroData):
        # --------- User-only fallback ---------
        if self.mode == 'U' or ('terminal' not in data.node_types and 'vm' not in data.node_types):
            user_x = self.user_proj(data['user'].x)
            return self.classifier(user_x)
        
        # 投影
        x = {
            'vm': self.vm_proj(data['vm'].x),
            'terminal': self.term_proj(data['terminal'].x),
            'user': self.user_proj(data['user'].x),
        }

        edge_index = data.edge_index_dict
        edge_attr = data.edge_attr_dict

        # 逐层 HeteroConv
        for conv in self.layers:
            updated_x = conv(x, edge_index, edge_attr)
            # updated部分只包括了接收消息的节点类型，并不是全部的节点类型
            # 因此需要把原来的字典和新的字典按键值合并，这样没更新的节点类型仍保持一致，更新的节点类型也正常更新了
            x = {**x, **updated_x}

        # 基于 user 节点分类
        out = self.classifier(x['user'])
        return out



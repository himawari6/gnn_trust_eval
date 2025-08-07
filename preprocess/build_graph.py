import torch
from torch_geometric.data import HeteroData
from typing import List, Dict

LABEL_MAP = {
    "允许访问": 0,
    "二次身份认证": 1,
    "限制访问": 2
}

def build_graph(sample: Dict) -> List[HeteroData]:
    """将一个样本字典转换为多个用户子图"""
    input_data = sample["input"]
    label_dict = sample["output"]["labels"]

    users = input_data["raw_users"]
    terminals = input_data["raw_terminals"]
    vms = input_data["raw_vms"]
    connections = input_data["connections"]

    terminals_indexed_by_id = {terminal['terminal_id']: terminal for terminal in terminals}
    vms_indexed_by_id = {vm['vm_id']: vm for vm in vms}

    # 按用户构图
    user_graphs = []
    for user in users:
        u_id = user["user_id"]

        # --- 构造当前用户子图 ---
        user_feat = torch.tensor([[
            int(user.get("userType", 0)),
            float(user.get("loginTotal", 0)),
            float(user.get("loginSucceed", 0)),
            float(user.get("ifLoginTimeOK", 0)),
            float(user.get("loginTimeBias") or 0.0),
            float(user.get("loginTimeDiff") or 0.0),
            float(user.get("ifIpAllow", 0)),
            float(user.get("ifAreaAllow") or 0.0),
        ]], dtype=torch.float)

        user_label = torch.tensor([LABEL_MAP[label_dict.get(u_id, "允许访问")]], dtype=torch.long)

        # 获取该用户所有连接
        u_conns = [c for c in connections if c["user_id"] == u_id]
        if not u_conns:
            data = HeteroData()
            data["user"].x = user_feat
            data["user"].y = user_label
            user_graphs.append(data)
            continue
        
        # 根据连接情况，列出参与建图的terminal, vm之id列表，并建立id和张量索引的关系
        involved_terminal_ids = set(c['terminal_id'] for c in connections)
        involved_vm_ids = set(c['vm_id'] for c in connections)
        terminal_id_index_map = {terminal_id: i for i, terminal_id in enumerate(involved_terminal_ids)}
        vm_id_index_map = {vm_id: i for i, vm_id in enumerate(involved_vm_ids)}

        # 根据上面的id名单正式导入terminal和vm特征
        # 和原本一样，这里产生索引号和输入特征的顺序都来源于相同的enumerate，因此可以保证顺序相同
        terminal_feats = []
        for tid in involved_terminal_ids:
            t = terminals_indexed_by_id[tid]
            terminal_feats.append([
                float(t.get("terminalType", 0)),
                float(t.get("userDiff", 0)),
            ])
        terminal_x = torch.tensor(terminal_feats, dtype=torch.float)

        vm_feats = []
        for vid in involved_vm_ids:
            v = vms_indexed_by_id[vid]
            vm_feats.append([
                float(v.get("VMOsAllow", 0)),
                float(v.get("VMOsVersionAllow", 0)),
                float(v.get("CPU", 0)),
                float(v.get("mem", 0)),
                float(v.get("VMConnectionUser", 0)),
                float(v.get("VMLoginTotal") or 0),
                float(v.get("VMLoginSucceed") or 0),
            ])
        vm_x = torch.tensor(vm_feats, dtype=torch.float)

        # terminal ↔ vm 边，直接引入连接的两个特征
        # 当然首先要把 terminal_id 和 vm_id 转换成输入张量中的 terminal 和 vm 索引号
        tv_edges = []
        tv_attrs = []
        for c in u_conns:
            t_id, v_id = c["terminal_id"], c["vm_id"]
            if t_id in terminal_id_index_map and v_id in vm_id_index_map:
                t_idx = terminal_id_index_map[t_id]
                v_idx = vm_id_index_map[v_id]
                tv_edges.append([t_idx, v_idx])
                tv_attrs.append([float(c.get("onlineTime") or 0), float(c.get("alertNum") or 0)])

        # user ↔ terminal 边（聚合），得到该用户对应不同终端的总连接时间和警报数
        ut_aggregated_feat = {}
        for c in u_conns:
            key = c["terminal_id"]
            ut_aggregated_feat.setdefault(key, {"onlineTime": 0, "alertNum": 0})
            ut_aggregated_feat[key]["onlineTime"] += float(c.get("onlineTime") or 0)
            ut_aggregated_feat[key]["alertNum"] += float(c.get("alertNum") or 0)

        # terminal ↔ vm 边，使用上面聚合出来的“新边”
        # 当然首先要把 user_id 和 terminal_id 转换成输入张量中的 user 和 terminal 索引号
        ut_edges = []
        ut_attrs = []
        for t_id, aggregated_feat in ut_aggregated_feat.items():
            if t_id not in terminal_id_index_map:
                continue
            t_idx = terminal_id_index_map[t_id]
            ut_edges.append([0, t_idx])  # 用户节点索引为0（单节点）
            ut_attrs.append([aggregated_feat["onlineTime"], aggregated_feat["alertNum"]])

        # --- 构造图 ---
        data = HeteroData()
        data["user"].x = user_feat
        data["user"].y = user_label
        data["terminal"].x = terminal_x
        data["vm"].x = vm_x

        data["user", "connects", "terminal"].edge_index = torch.tensor(ut_edges, dtype=torch.long).t().contiguous()
        data["user", "connects", "terminal"].edge_attr = torch.tensor(ut_attrs, dtype=torch.float)

        data["terminal", "connects", "vm"].edge_index = torch.tensor(tv_edges, dtype=torch.long).t().contiguous()
        data["terminal", "connects", "vm"].edge_attr = torch.tensor(tv_attrs, dtype=torch.float)

        user_graphs.append(data)

    return user_graphs

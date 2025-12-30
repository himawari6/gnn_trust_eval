import os
import torch
from torch_geometric.data import HeteroData, Data
from typing import List

NODE_TYPE_MAP = {
    'user': 0,
    'terminal': 1,
    'vm': 2
}

def hetero_list_to_homo_list(
    hetero_list: List[HeteroData],
    add_type_onehot: bool = True,
    type_num = len(NODE_TYPE_MAP)
) -> List[Data]:
    """
    将 List[HeteroData] 转换为 List[Data]（同构图）

    每个 Data 包含：
    - x（拼接类型 one-hot）
    - edge_index
    - edge_attr
    - y（仅 user 节点有效，其余为 -1）
    - user_mask
    """

    homo_list = []
    node_types = NODE_TYPE_MAP.keys()

    for hetero in hetero_list:
        # 1. 转为同构图
        homo = hetero.to_homogeneous()
        assert 'user' in hetero.node_types, "HeteroData 中必须包含 user 节点"

        # 2. 生成指示每个节点类型的node_type_idx数组
        node_type_idx = torch.empty_like(homo.node_type)
        for name, idx in NODE_TYPE_MAP.items():
            # 获取在该图中，to_homo给该类型自动映射到了什么数值，记为该类型的本地映射
            # 如果该图中不存在这一节点类型，就把本地映射记成-1
            local_idx = hetero.node_types.index(name) if name in hetero.node_types else -1
            # 把所有的本地映射值替换为全局映射
            # if指的是“该类型的本地映射不是-1，所以该图中存在这一节点类型”
            # 中括号里是布尔批量索引，能指示出全部的应修改位置
            if local_idx >= 0: 
                node_type_idx[homo.node_type == local_idx] = idx

        # 3. 构造 user_mask
        user_mask = (homo.node_type == NODE_TYPE_MAP['user'])
        homo.user_mask = user_mask

        # 4. 构造 y（对齐到所有节点，首先将所有节点的y设为-1）
        y = torch.full(
            (homo.num_nodes,),
            -1,
            dtype=torch.long
        )
        # 修改原类型为user的节点，令其y恢复原值，其他的还是-1
        y[user_mask] = hetero['user'].y
        homo.y = y

        # 5. 添加节点类型 one-hot
        if add_type_onehot:
            type_onehot = torch.nn.functional.one_hot(
                node_type_idx,
                num_classes=type_num
            ).float()
            homo.x = torch.cat([homo.x, type_onehot], dim=1)

        # 6. 清理不再需要的属性（可选，但推荐）
        homo.node_type = None
        homo.edge_type = None

        homo_list.append(homo)

    return homo_list


def convert_pt_file(
    input_pt_path: str,
    output_pt_path: str,
    add_type_onehot: bool = True
):
    """
    读取 .pt (List[HeteroData])，转换为同构图 List[Data]，并保存
    """

    if not os.path.exists(input_pt_path):
        raise FileNotFoundError(f"输入文件不存在：{input_pt_path}")

    hetero_list = torch.load(input_pt_path)
    assert isinstance(hetero_list, list), "pt 文件必须是 List[HeteroData]"

    homo_list = hetero_list_to_homo_list(
        hetero_list,
        add_type_onehot=add_type_onehot
    )

    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save(homo_list, output_pt_path)

    print(f"转换完成：")
    print(f"  输入: {input_pt_path}")
    print(f"  输出: {output_pt_path}")
    print(f"  样本数: {len(homo_list)}")


if __name__ == "__main__":
    convert_pt_file(
        input_pt_path="data/graph/evaluate/evaluate_samples.pt",
        output_pt_path="data/graph/evaluate/evaluate_samples_homo.pt",
        add_type_onehot=True
    )

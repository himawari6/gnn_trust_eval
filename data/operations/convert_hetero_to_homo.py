import os
import torch
from torch_geometric.data import HeteroData, Data
from typing import List


def hetero_list_to_homo_list(
    hetero_list: List[HeteroData],
    add_type_onehot: bool = True
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

    for hetero in hetero_list:
        # 1. 转为同构图
        homo = hetero.to_homogeneous()

        # 2. 确定 user 类型 index
        node_types = hetero.node_types
        assert 'user' in node_types, "HeteroData 中必须包含 user 节点"

        user_type_idx = node_types.index('user')

        # 3. 构造 user_mask
        user_mask = (homo.node_type == user_type_idx)
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
            num_types = len(node_types)
            type_onehot = torch.nn.functional.one_hot(
                homo.node_type,
                num_classes=num_types
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
    # 示例用法
    convert_pt_file(
        input_pt_path="data/graph/evaluate/evaluate_samples.pt",
        output_pt_path="data/graph/evaluate/evaluate_samples_homo.pt",
        add_type_onehot=True
    )

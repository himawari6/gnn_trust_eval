import argparse
from preprocess.build_graph import build_and_save_graph
from preprocess.build_graph_with_alert_on_node import build_and_save_graph_with_alert_on_node
from preprocess.build_graph_with_vm2user_edge import build_and_save_graph_with_vm2user_edge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="data\sample\merged\merged_sample_20250816_164358.json", help="路径：原始 JSON 样本文件")
    parser.add_argument("--output", type=str, default="data/graph/sample_2.pt", help="输出文件")
    # parser.add_argument("--alert-on-edges", type=str, default='y', help='告警信息是否在边上')
    # parser.add_argument("--uv-edges", type=str, default='n', help="是否构建具有虚拟机->用户边的图")
    parser.add_argument("--mode", type=str, choices=["normal", "alert_on_nodes", "vm2user"], required=True,
        help="执行模式：normal=正常, alert_on_nodes=告警在节点上, vm2user=有虚拟机->用户边")
    args = parser.parse_args()

    if args.mode == "vm2user":
        build_and_save_graph_with_vm2user_edge(args.sample, args.output)
    elif args.mode == "normal":
        build_and_save_graph(args.sample, args.output)
    elif args.mode == "alert_on_nodes":
        build_and_save_graph_with_alert_on_node(args.sample, args.output)
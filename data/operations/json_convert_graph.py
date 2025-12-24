import argparse
from preprocess.save_graph import build_and_save_graph
from preprocess.build_save_graph_with_alert_on_node import build_and_save_graph_with_alert_on_node

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="data\sample\merged\merged_sample_20250816_164358.json", help="路径：原始 JSON 样本文件")
    parser.add_argument("--output", type=str, default="data/graph/sample_2.pt", help="输出文件")
    parser.add_argument("--alert-on-edges", type=str, default='y', help='告警信息是否在边上')
    args = parser.parse_args()

    if args.alert_on_edges == 'y':
        build_and_save_graph(args.sample, args.output)

    elif args.alert_on_edges == 'n':
        build_and_save_graph_with_alert_on_node(args.sample, args.output)
import argparse
from preprocess.save_graph import build_and_save_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True, help="路径：原始 JSON 样本文件")
    parser.add_argument("--output", type=str, default="data/graph/sample_1.pt", help="输出文件")
    args = parser.parse_args()

    build_and_save_graph(args.sample, args.output)
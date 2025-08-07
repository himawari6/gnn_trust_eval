import json
import torch
from preprocess.build_graph import build_graph

def build_and_save_graph(sample_path: str, save_path: str):
    with open(sample_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    graphs_list = []
    for sample in samples:
        graphs = build_graph(sample)
        graphs_list.extend(graphs)

    torch.save(graphs_list, save_path)


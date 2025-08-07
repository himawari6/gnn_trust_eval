import json
from preprocess.build_graph import build_graph

# 读取你的样本 JSON 文件（包含样本列表）
with open("data/sample/sample_20250807_214937.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# 只取第一条样本作为 demo
hetero_data = build_graph(samples[0])

print(hetero_data)
print(hetero_data[1]['user'])
print(hetero_data[1]['terminal'])
print(hetero_data[1]['vm'])
print(hetero_data[1]['user', 'connects', 'terminal'])
print(hetero_data[1]['terminal', 'connects', 'vm'])
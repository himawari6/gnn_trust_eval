import json
from datetime import datetime

def merge_json_samples(samples_path: list[str], output_path: str):
    samples_list = []
    for sample_path in samples_path:
        with open(sample_path, 'r', encoding="utf-8") as f:
            sample_list = json.load(f)
        samples_list.extend(sample_list)
    
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(samples_list, f, ensure_ascii=False, indent=4)
    
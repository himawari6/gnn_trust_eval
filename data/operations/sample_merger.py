import json
import os
from datetime import datetime

def merge_json_samples(samples_path: list[str], output_path: str):
    samples_list = []
    for sample_path in samples_path:
        with open(sample_path, 'r', encoding="utf-8") as f:
            sample_list = json.load(f)
        samples_list.extend(sample_list)
    
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(samples_list, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    samples_dir = 'data/sample'
    samples_path_list = []
    for entry in os.scandir(samples_dir):
        if entry.is_file():
            f_path = entry.path
            samples_path_list.append(f_path)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = samples_dir + '/merged' + f'/merged_sample_{current_time}.json'

    merge_json_samples(samples_path=samples_path_list, output_path=output_path)

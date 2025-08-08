import os
from datetime import datetime
from data.operations.sample_merger import merge_json_samples

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

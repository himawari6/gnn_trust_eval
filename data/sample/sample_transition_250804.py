import json

def simplify_sample_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    for sample in samples:
        # input
        del sample['input']['time'], sample['input']['level'], sample['input']['message'], sample['input']['type']
        # output
        del sample['output']['time'], sample['output']['level'], sample['output']['message'], sample['output']['type']
        simplified_results = {
            user_id: result_dict["label"]
            for user_id, result_dict in sample.get("output", {}).get("results", {}).items()
        }
        sample['output']["labels"] = simplified_results
        del sample['output']['results']

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)

# 示例用法
simplify_sample_json(
    input_path="sample_01.json",
    output_path="sample_1.json"
)
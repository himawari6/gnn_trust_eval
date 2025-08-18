import torch
import random

random.seed('?')

dataset_path = 'data\graph\merged_sample_20250816_164358.pt'

data_list = torch.load(dataset_path)

print(f"总样本数: {len(data_list)}")

indices = list(range(len(data_list)))
random.shuffle(indices)

split = int(0.8 * len(indices))
train_idx = indices[:split]
test_idx = indices[split:]

train_data = [data_list[i] for i in train_idx]
test_data = [data_list[i] for i in test_idx]

print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")

torch.save(train_data, "data\\graph\\train_samples.pt")
torch.save(test_data, "data\\graph\\test_samples.pt")
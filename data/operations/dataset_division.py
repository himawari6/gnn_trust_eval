import torch
import random
import argparse

def dataset_division(dataset_path, train_dataset_path, evaluate_dataset_path):
    random.seed('?')

    # dataset_path = 'data\\graph\\main\\full_sample_with_vm2user_edge.pt'
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

    # torch.save(train_data, "data\\graph\\train_samples_with_vm2user_edge.pt")
    # torch.save(test_data, "data\\graph\\evaluate_samples_with_vm2user_edge.pt")
    torch.save(train_data, train_dataset_path)
    torch.save(test_data, evaluate_dataset_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='data\\graph\\main\\full_sample.pt')
    parser.add_argument("--train-data-path", type=str, default='data\\graph\\train\\train_samples.pt')
    parser.add_argument("--test-data-path", type=str, default='data\\graph\\evaluate\\evaluate_samples.pt')
    args = parser.parse_args()

    dataset_division(dataset_path=args.data_path, 
                     train_dataset_path=args.train_data_path, 
                     evaluate_dataset_path=args.test_data_path)
import torch

all_graphs = torch.load("data\graph\merged_sample_20250816_164358.pt")
data = all_graphs[7]
save_path = "data\\graph\\toy.pt"
torch.save(data, save_path)
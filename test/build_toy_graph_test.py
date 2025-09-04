import torch

all_graphs = torch.load("data\\graph\\full_sample.pt")
data = all_graphs[7]
save_path = "data\\graph\\toy.pt"
torch.save(data, save_path)
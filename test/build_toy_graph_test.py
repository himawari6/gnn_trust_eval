import torch

all_graphs = torch.load("data\\graph\\test.pt")
data = all_graphs[457]
save_path = "data\\graph\\toy_alert_on_node.pt"
torch.save(data, save_path)
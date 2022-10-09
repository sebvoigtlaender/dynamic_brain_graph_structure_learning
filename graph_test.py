import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')


data = dataset[0]  # Get the first graph object.


torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(train_dataset[1].edge_index)

# print(f'Number of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}')

# from torch_geometric.loader import DataLoader

# train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data.edge_index)
#     print()
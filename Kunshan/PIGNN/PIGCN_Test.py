from sympy.physics.units import length
from visdom import Visdom
from PIGCN_Net import GCN_Model, GCN_Dataset
import torch
from torch.utils.data import DataLoader
from Function import nse_loss
from visdom import Visdom
import numpy as np
import pandas as pd
import time
from Function import Area, subsets, subsets_increasing, subsets_decreasing
import matplotlib.pyplot as plt

# region Best_params
space_lr = [1e-5, 1e-4, 1e-3]
space_weight_decay = [1e-7, 1e-6, 1e-5]
space_edge_hop = [2, 3, 4]
space_node_hop = [2, 3, 4]
lr = [space_lr[0], space_lr[2]]
weight_decay = [space_weight_decay[0], space_weight_decay[2]]
hops_edge = space_edge_hop[1]
hops_node = space_node_hop[1]
# region Parameter import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

dev_size = batch_size
epochs = 100
train_node_index = [2, 8, 11, 17]
dev_node_index = [2, 5, 8, 11, 14, 17]
train_edge_index = [14]
dev_edge_index = [5, 14]
length_index = [15, 15, 9, 16, 16, 11, 17, 17, 12, 17, 17, 13, 18, 18, 14, 19, 19, 16]
length_index = torch.tensor(list(map(int, length_index))).to(device)
# length_index = length_index * 100
max_length = torch.max(length_index).to(device)
# resolution = 1 / 100
resolution = 1
# Instantiate model and run forward pass
node_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.long).to(device)
edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18]],
                          dtype=torch.long).t().contiguous().to(device)

num_edges = edge_index.shape[1]
num_nodes = len(node_index)
## Pipe physical properties
D_input = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
n_input= torch.tensor([0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009])
S0_input = torch.tensor([0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])

## Initialize adjacency matrices with zeros
adj_matrix = torch.zeros(num_edges, num_nodes)  # edge-node relation
node_matrix = torch.zeros(num_nodes, num_nodes)  # node-node relation
edge_matrix = torch.zeros(num_edges, num_edges)  # edge-edge relation
# Fill edge-node adjacency matrix
for i in range(num_edges):
    adj_matrix[i, edge_index[0, i]] = 1
    adj_matrix[i, edge_index[1, i]] = 1
adj_matrix = adj_matrix.to(device)

# Fill node-node adjacency matrix
node_matrix[edge_index[0], edge_index[1]] = 1
node_matrix[edge_index[1], edge_index[0]] = 1
node_matrix[edge_index[0], edge_index[0]] = 1
node_matrix[edge_index[1], edge_index[1]] = 1
node_matrix = node_matrix.to(device)

# Fill edge-edge adjacency matrix
for i in range(num_edges):
    for j in range(i, num_edges):
        start_i, end_i = edge_index[:, i]
        start_j, end_j = edge_index[:, j]
        if start_i == start_j or start_i == end_j or end_i == start_j or end_i == end_j:
            edge_matrix[i, j] = 1
            edge_matrix[j, i] = 1
edge_matrix = edge_matrix.to(device)

# Load data
train_dataset = GCN_Dataset(edge_index, length_index, file_path='')
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)
test_dataset = GCN_Dataset(edge_index, length_index, file_path='')
devloader = DataLoader(test_dataset, batch_size=dev_size, shuffle=False, drop_last=False, pin_memory=False)
# endregion
# endregion

# region Parameter import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev_size = 128
epochs = 1000
train_node_index = [2, 5, 8, 11, 14, 17]
train_edge_index = [5, 14]
length_index = [15, 15, 9, 16, 16, 11, 17, 17, 12, 17, 17, 13, 18, 18, 14, 19, 19, 16]
length_index = torch.tensor(list(map(int, length_index))).to(device)
length_index = length_index
max_length = torch.max(length_index).to(device)
n = max_length
# Instantiate model and run forward pass
node_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ,18], dtype=torch.long).to(device)
edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18]],
                          dtype=torch.long).t().contiguous().to(device)

num_edges = edge_index.shape[1]
num_nodes = len(node_index)
## Pipe physical properties
D_input = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
n_input= torch.tensor([0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013])
S0_input = torch.tensor([0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])

## Initialize adjacency matrices with zeros
adj_matrix = torch.zeros(num_edges, num_nodes)  # edge-node relation
node_matrix = torch.zeros(num_nodes, num_nodes)  # node-node relation
edge_matrix = torch.zeros(num_edges, num_edges)  # edge-edge relation
# Fill edge-node adjacency matrix
for i in range(num_edges):
    adj_matrix[i, edge_index[0, i]] = 1
    adj_matrix[i, edge_index[1, i]] = 1
adj_matrix = adj_matrix.to(device)

# Fill node-node adjacency matrix
node_matrix[edge_index[0], edge_index[1]] = 1
node_matrix[edge_index[1], edge_index[0]] = 1
node_matrix[edge_index[0], edge_index[0]] = 1
node_matrix[edge_index[1], edge_index[1]] = 1
node_matrix = node_matrix.to(device)

# Fill edge-edge adjacency matrix
for i in range(num_edges):
    for j in range(i, num_edges):
        start_i, end_i = edge_index[:, i]
        start_j, end_j = edge_index[:, j]
        if start_i == start_j or start_i == end_j or end_i == start_j or end_i == end_j:
            edge_matrix[i, j] = 1
            edge_matrix[j, i] = 1
edge_matrix = edge_matrix.to(device)

# Load data
test_dataset = GCN_Dataset(edge_index, length_index, file_path='')
devloader = DataLoader(test_dataset, batch_size=dev_size, shuffle=False, drop_last=False)
# endregion

model = GCN_Model(resolution=resolution, hops_node=hops_node,
                  forward_node=[64, 128, 256, 128, 64], forward_edge=[64, 128, 256, 128, 64],
                  transcov_hidden=[64, 128, 256], transcov_edge=[32, 64, 32],
                  length_index=length_index,
                  hops_edge=hops_edge, adj_matrix=adj_matrix,
                  node_index=node_index, node_matrix=node_matrix,
                  edge_matrix=edge_matrix, edge_index=edge_index).to(device)

model = model.to(device)
model.eval()
mask_node = []
mask_edge = []
max_length = torch.max(length_index).to(device)
num_edges = len(length_index)
D_feature_dev = torch.zeros((num_edges, max_length))
D_feature_dev = D_feature_dev + D_input.unsqueeze(dim=1)
D_feature_dev = D_feature_dev.unsqueeze(0).repeat(dev_size, 1, 1)
D_feature_dev = D_feature_dev.to(device)

D_input = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

S0_feature = torch.zeros((num_edges, max_length))
S0_feature = S0_feature + S0_input.unsqueeze(dim=1)
S0_feature = S0_feature.unsqueeze(0).repeat(dev_size, 1, 1)
S0_feature = S0_feature.to(device)

n_feature = torch.zeros((num_edges, max_length))
n_feature = n_feature + n_input.unsqueeze(dim=1)
n_feature = n_feature.unsqueeze(0).repeat(dev_size, 1, 1)
n_feature = n_feature.to(device)
g = 9.8

# One-step prediction -- Dataset
# viz = Visdom(env='1130_Test_epoch1000_Single')
# viz = Visdom(env='1209_PIGCN_Best_Test')

max_nse_all = -9999
max_epoch = 0

model.load_state_dict(torch.load(''))

node_out = np.zeros((863, num_nodes))
edge_out = np.zeros((863, num_edges, 2))
edge_full_out = np.zeros((863, 279, 301, 2))
output_node_feature_all = torch.zeros((863, num_nodes)).to(device)
output_edge_feature_all = torch.zeros((863, num_edges, 2)).to(device)

for index, data in enumerate(devloader, 0):
    start_time = time.time()
    node_feature, edge_feature, output_node_feature, output_edge_feature = data
    node_feature = node_feature.to(device)
    edge_feature = edge_feature.to(device)
    output_node_feature = output_node_feature.to(device)
    output_edge_feature = output_edge_feature.to(device)
    node_updated, edge_updated, edge_full_updated, lambda1, lambda2, lambda3 = model(node_feature, edge_feature, D_input)
    node_out[index * dev_size:(index + 1) * dev_size, :] = node_updated.detach().cpu().numpy()
    edge_out[index * dev_size:(index + 1) * dev_size, :, :] = edge_updated.detach().cpu().numpy()
    edge_full_out[index * dev_size:(index + 1) * dev_size, :, :, :] = edge_full_updated.detach().cpu().numpy()
    output_node_feature_all[index * dev_size:(index + 1) * dev_size, :] = output_node_feature.detach().cpu()
    output_edge_feature_all[index * dev_size:(index + 1) * dev_size, :, :] = output_edge_feature.detach().cpu()
print('Predicting Time: ', time.time() - start_time)

# full_w_out = np.zeros((279, 863 * 300))
# full_u_out = np.zeros((279, 863 * 300))
# from tqdm import tqdm
# for i in tqdm(range(863)):
#     full_w_out[:, i * 300:(i + 1) * 300] = edge_full_out[i, :, 1:, 0]
#     full_u_out[:, i * 300:(i + 1) * 300] = edge_full_out[i, :, 1:, 1]
dev_node = [2, 5, 8, 11, 14, 17]
dev_node = [i - 1 for i in dev_node]
dev_links = [5, 14]
dev_links = [i - 1 for i in dev_links]
output_node_feature_all = output_node_feature_all.cpu()
output_edge_feature_all = output_edge_feature_all.cpu()
node_out = torch.tensor(node_out)
edge_out = torch.tensor(edge_out)
for i in range(len(dev_node)):
    print(nse_loss(output_node_feature_all[:, dev_node[i]], node_out[:, dev_node[i]]).item())
for i in range(len(dev_links)):
    print(nse_loss(output_edge_feature_all[:, dev_links[i], 0], edge_out[:, dev_links[i], 0]).item())
# import h5py
# # Create HDF5 file and write data
node_out = pd.DataFrame(node_out)
node_out.to_csv('node_level.csv', index=False)
w_out = pd.DataFrame(edge_out[:, :, 0])
w_out.to_csv('edge_capacity.csv', index=False)
u_out = pd.DataFrame(edge_out[:, :, 1])
u_out.to_csv('edge_velocity.csv', index=False)


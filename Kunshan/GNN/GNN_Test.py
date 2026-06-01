from visdom import Visdom
from GNN_Net import GNN_Model, GNN_Dataset
import torch
from torch.utils.data import DataLoader
from Function import nse_loss
import numpy as np
import pandas as pd
import time
from Function import Area, subsets, subsets_increasing, subsets_decreasing
import matplotlib.pyplot as plt
start_time = time.time()
# region Best_params
space_node = subsets([8, 16, 32])
space_edge = subsets([32, 64, 128])
space_node2edge = subsets_increasing([32, 64, 128])
space_edge2node = subsets_decreasing([32, 64, 128])
space_lr = [1e-5, 1e-4, 1e-3]
space_weight_decay = [1e-7, 1e-6, 1e-5]
space_edge_hop = [2, 3, 4]
space_node_hop = [2, 3, 4]
lr = space_lr[2]
weight_decay = space_weight_decay[0]
hidden_kernels_node = space_node[4]
hidden_kernels_edge = space_edge[6]
hidden_kernels_node2edge = space_node2edge[2]
hidden_kernels_edge2node = space_edge2node[3]
hops_edge = space_edge_hop[0]
hops_node = space_node_hop[2]
print(hidden_kernels_node)
print(hidden_kernels_edge)
print(hidden_kernels_node2edge)
print(hidden_kernels_edge2node)
start_time = time.time()
# endregion

# region Parameter import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev_size = 863
epochs = 1000
pred_node_index = [2, 5, 8, 11, 14, 17]
pred_edge_index = [5, 14]
length_index = [15, 15, 9, 16, 16, 11, 17, 17, 12, 17, 17, 13, 18, 18, 14, 19, 19, 16]
length_index = torch.tensor(list(map(int, length_index))).to(device)
length_index = length_index * 100
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
test_dataset = GNN_Dataset(edge_index, file_path='')
devloader = DataLoader(test_dataset, batch_size=dev_size, shuffle=False, drop_last=True)
# endregion

model = GNN_Model(hops_node=hops_node, hidden_kernels_node=hidden_kernels_node,
                  hidden_kernels_node2edge=hidden_kernels_node2edge,
                  hops_edge=hops_edge, hidden_kernels_edge=hidden_kernels_edge,
                  hidden_kernels_edge2node=hidden_kernels_edge2node,
                  adj_matrix=adj_matrix, node_index=node_index, node_matrix=node_matrix,
                  edge_matrix=edge_matrix, edge_index=edge_index).to(device)

model = model.to(device)
model.eval()
mask_node = [4, 13]
mask_edge = [4]
max_length = torch.max(length_index).to(device)
num_edges = len(length_index)
D_feature_dev = torch.zeros((num_edges, 1))
D_feature_dev = D_feature_dev + D_input.unsqueeze(dim=1)
D_feature_dev = D_feature_dev.unsqueeze(0).repeat(dev_size, 1, 1)
D_feature_dev = D_feature_dev.to(device)

S0_feature = torch.zeros((num_edges, 1))
S0_feature = S0_feature + S0_input.unsqueeze(dim=1)
S0_feature = S0_feature.unsqueeze(0).repeat(dev_size, 1, 1)
S0_feature = S0_feature.to(device)

n_feature = torch.zeros((num_edges, 1))
n_feature = n_feature + n_input.unsqueeze(dim=1)
n_feature = n_feature.unsqueeze(0).repeat(dev_size, 1, 1)
n_feature = n_feature.to(device)
g = 9.8

# One-step prediction -- Dataset
viz = Visdom(env='GNN_Best_Test')
model.load_state_dict(torch.load(''))
for index, data in enumerate(devloader, 0):

    node_feature, edge_feature, output_node_feature, output_edge_feature = data
    node_feature = node_feature.to(device)
    edge_feature = edge_feature.to(device)
    output_node_feature = output_node_feature.to(device)
    output_edge_feature = output_edge_feature.to(device)
    for k in mask_node:
        node_feature[:, k, :] = 0
    for k in mask_edge:
        edge_feature[:, k, :] = 0
    node_updated, edge_updated, lambda1 = model(node_feature, edge_feature)
    edge_updated_w = edge_updated[:, :, 0]
    edge_updated_u = edge_updated[:, :, 1]

    mean_update_w = edge_updated_w
    mean_update_u = edge_updated_u
    mean_update_w_1m = edge_updated_w.cpu().detach().numpy()
    mean_update_u_1m = edge_updated_u.cpu().detach().numpy()

    mean_update_w_1m = pd.DataFrame(mean_update_w_1m)
    mean_update_u_1m = pd.DataFrame(mean_update_u_1m)
    mean_update_w_1m.to_csv('Predict_Capacity_GNN.csv', index=False)
    mean_update_u_1m.to_csv('Predict_Velocity_GNN.csv', index=False)

    for i in range(node_index.size(0)):
        title = node_index[i].item()
        viz.line(node_updated[:, node_index[i]].cpu().detach().numpy(), [time for time in range(dev_size)], win='NODE_'+str(title),
                 name='Pred', update='append', opts=dict(title='NODE_'+str(title)))
        viz.line(output_node_feature[:, node_index[i]].cpu().detach().numpy(), [time for time in range(dev_size)], win='NODE_'+str(title),
                 name='True', update='append', opts=dict(title='NODE_'+str(title)))
        print('NSE_NODE_'+str(title)+":  ", nse_loss(y_pred=node_updated[:, node_index[i]], y_true=output_node_feature[:, node_index[i]]).cpu().detach().numpy())
        print('MAE_NODE_' + str(title) + ":  ", torch.mean(abs(node_updated[:, node_index[i]] - output_node_feature[:, node_index[i]])))
    for j in range(num_edges):
        title = str(edge_index[0, j].item()) + '-' + str(edge_index[1, j].item())
        viz.line(mean_update_w[:, j].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_Capacity_'+title,
                 name='Pred', update='append', opts=dict(title='EDGE_Capacity_'+title))
        viz.line(output_edge_feature[:, j, 0].cpu().detach().numpy(), [time for time in range(dev_size)],
                 win='EDGE_Capacity_'+title,
                 name='True', update='append', opts=dict(title='EDGE_Capacity_'+title))
        viz.line(mean_update_u[:, j].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_Velocity_'+title,
                 name='Pred', update='append', opts=dict(title='EDGE_Velocity_'+title))
        viz.line(output_edge_feature[:, j, 1].cpu().detach().numpy(), [time for time in range(dev_size)],
                 win='EDGE_Velocity_'+title,
                 name='True', update='append', opts=dict(title='EDGE_Velocity_'+title))
        print('NSE_EDGE_Capacity_'+title+":  ", nse_loss(y_pred=mean_update_w[:, j], y_true=output_edge_feature[:, j, 0]).cpu().detach().numpy())
        print('NSE_EDGE_Velocity_'+title+":  ", nse_loss(y_pred=mean_update_u[:, j], y_true=output_edge_feature[:, j, 1]).cpu().detach().numpy())
        print('MAE_EDGE_Capacity_' + title + ":  ",
              torch.mean(abs(mean_update_w[:, j] - output_edge_feature[:, j, 0])))
        print('MAE_EDGE_Velocity_' + title + ":  ",
              torch.mean(abs(mean_update_u[:, j] - output_edge_feature[:, j, 1])))
        # viz.surf(edge_updated_w[:, j, :length_index[j]].cpu().detach().numpy(), win='Heatmap_Capacity_'+title, opts=dict(title='Heatmap_Capacity_'+title))
        # viz.surf(edge_updated_u[:, j, :length_index[j]].cpu().detach().numpy(), win='Heatmap_Velocity_'+title, opts=dict(title='Heatmap_Velocity_'+title))

    print(nse_loss(y_pred=node_updated, y_true=output_node_feature).cpu().detach().numpy())
    print(nse_loss(y_pred=mean_update_w, y_true=output_edge_feature[:, :, 0]).cpu().detach().numpy())
    print(nse_loss(y_pred=mean_update_u, y_true=output_edge_feature[:, :, 1]).cpu().detach().numpy())
    node_nse = nse_loss(y_pred=node_updated, y_true=output_node_feature).cpu().detach().numpy()
    node_updated = node_updated.cpu().detach().numpy()
    node_updated = pd.DataFrame(node_updated)
    node_updated.to_csv('Predict_Node_GNN.csv', index=False)
end_time = time.time()
print('Time: ', end_time - start_time)

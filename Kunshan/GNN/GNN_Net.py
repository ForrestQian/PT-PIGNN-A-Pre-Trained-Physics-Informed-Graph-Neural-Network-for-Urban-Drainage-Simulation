import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Function import Area
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GNN dataset
class GNN_Dataset(Dataset):
    def __init__(self, edge_index, file_path):

        self.file_path = file_path
        self.edge_index = edge_index
        self.node_data_h = pd.read_excel(self.file_path, sheet_name='Node_Depth')
        self.node_data_h = np.array(self.node_data_h)[:, 1:]

        self.node_data_q = pd.read_excel(self.file_path, sheet_name='Node_Inflow')
        self.node_data_q = np.array(self.node_data_q)[:, 1:]

        self.edge_data_capacity = pd.read_excel(self.file_path, sheet_name='Link_Flow')
        self.edge_data_capacity = np.array(self.edge_data_capacity)[:, 1:]

        self.edge_data_velocity = pd.read_excel(self.file_path, sheet_name='Link_Velocity')
        self.edge_data_velocity = np.array(self.edge_data_velocity)[:, 1:]
    def __len__(self):
        return (self.node_data_h.shape[0] - 1)
    def __getitem__(self, index):
        # Load node water level and inflow data
        node_feature_h = self.node_data_h[index, :]
        node_feature_h = [float(element) for element in node_feature_h]
        node_feature_h = torch.tensor(node_feature_h).unsqueeze(dim=1)

        node_feature_q = self.node_data_q[index, :]
        node_feature_q = [float(element) for element in node_feature_q]
        node_feature_q = torch.tensor(node_feature_q).unsqueeze(dim=1)

        node_feature = torch.cat((node_feature_h, node_feature_q), dim=1)
        # Load pipe average occupancy (converted to wetted area) and average velocity
        edge_input_capacity = self.edge_data_capacity[index, :]
        edge_input_capacity = [float(element) for element in edge_input_capacity]
        edge_input_capacity = torch.tensor(edge_input_capacity)

        edge_input_velocity = self.edge_data_velocity[index, :]
        edge_input_velocity = [float(element) for element in edge_input_velocity]
        edge_input_velocity = torch.tensor(edge_input_velocity)

        # Build pipe feature matrix: m pipes x 2 features x L max length (zero-padded if shorter)
        edge_feature = torch.zeros((self.edge_index.shape[1], 2))

        output_node_feature = self.node_data_h[index + 1, :]
        output_node_feature = [float(element) for element in output_node_feature]
        output_node_feature = torch.tensor(output_node_feature)

        output_edge_capacity = self.edge_data_capacity[index + 1, :]
        output_edge_capacity = [float(element) for element in output_edge_capacity]
        output_edge_capacity = torch.tensor(output_edge_capacity).unsqueeze(dim=1)

        output_edge_velocity = self.edge_data_velocity[index + 1, :]
        output_edge_velocity = [float(element) for element in output_edge_velocity]
        output_edge_velocity = torch.tensor(output_edge_velocity).unsqueeze(dim=1)

        output_edge_feature = torch.cat((output_edge_capacity, output_edge_velocity), dim=1)

        for j in range(self.edge_index.shape[1]):
            edge_feature[j, 0] = edge_input_capacity[j]
            edge_feature[j, 1] = edge_input_velocity[j]
        return [node_feature, edge_feature, output_node_feature, output_edge_feature]

# Compute extended adjacency matrix for nodes (multi-hop connections)
def compute_extended_adjacency(node_matrix, hops):
    extended_node_matrix = node_matrix.clone()
    power_matrix = node_matrix.clone()
    for _ in range(hops - 1):  # accumulate multi-hop adjacency
        power_matrix = torch.matmul(power_matrix, node_matrix)
        extended_node_matrix += power_matrix
    # Binarize adjacency matrix to avoid values exceeding range after accumulation
    extended_node_matrix = (extended_node_matrix > 0).float()
    return extended_node_matrix

def compute_adjacency_matrix(edge_matrix, hops):
    extended_edge_matrix = edge_matrix.clone()
    power_matrix = edge_matrix.clone()
    for _ in range(hops - 1):  # accumulate multi-hop adjacency
        power_matrix = torch.matmul(power_matrix, edge_matrix)
        extended_edge_matrix += power_matrix
    # Binarize adjacency matrix to avoid values exceeding range after accumulation
    extended_edge_matrix = (extended_edge_matrix > 0).float()
    return extended_edge_matrix

# GNN model
class GNN_Model(nn.Module):
    def __init__(self, hops_node, hops_edge,
                 hidden_kernels_node, hidden_kernels_edge,
                 hidden_kernels_node2edge, hidden_kernels_edge2node,
                 adj_matrix, node_index, node_matrix, edge_matrix, edge_index):
        super(GNN_Model, self).__init__()
        # Project node features to the same dim as edge features
        self.node_to_edge_projection = nn.ModuleList()
        if hidden_kernels_node2edge is not None:
            pre_neurons = 2
            for neurons in hidden_kernels_node2edge:
                self.node_to_edge_projection.append(nn.Linear(pre_neurons, neurons))
                pre_neurons = neurons
            self.node_to_edge_projection.append(nn.Linear(pre_neurons, 2))
        else:
            self.node_to_edge_projection = nn.Linear(2, 2)
        # Project edge features to node feature dim
        self.edge_to_node_projection = nn.ModuleList()
        if hidden_kernels_edge2node is not None:
            pre_neurons = 2
            for neurons in hidden_kernels_edge2node:
                self.edge_to_node_projection.append(nn.Linear(pre_neurons, neurons))
                pre_neurons = neurons
            self.edge_to_node_projection.append(nn.Linear(pre_neurons, 2))
        else:
            self.edge_to_node_projection = nn.Linear(2, 2) # (edge * 1)
        # Node update
        self.node_update = nn.ModuleList()
        if hidden_kernels_node is not None:
            pre_neurons = 2
            for neurons in hidden_kernels_node:
                self.node_update.append(nn.Linear(pre_neurons, neurons))
                pre_neurons = neurons
            self.node_update.append(nn.Linear(pre_neurons, 1))
        else:
            self.node_update = nn.Linear(2, 1)
        # Edge update
        self.edge_update = nn.ModuleList()
        if hidden_kernels_edge is not None:
            pre_neurons = 2
            for neurons in hidden_kernels_edge:
                self.edge_update.append(nn.Linear(pre_neurons, neurons))
                pre_neurons = neurons
            self.edge_update.append(nn.Linear(pre_neurons, 2))
        else:
            self.edge_update = nn.Linear(2, 2)

        self.lambda1 = torch.nn.Parameter(torch.tensor([0.50]))
        self.hops_node = hops_node
        self.hops_edge = hops_edge
        self.adj_matrix = adj_matrix
        self.node_index = node_index
        self.edge_matrix = edge_matrix
        self.edge_index = edge_index
        self.node_matrix = node_matrix
        self.layer_node = len(self.node_update)
        self.layer_edge = len(self.edge_update)
        self.layer_node2edge = len(self.node_to_edge_projection)
        self.layer_edge2node = len(self.edge_to_node_projection)

    def forward(self, h, e):
        lambda1 = torch.clamp(self.lambda1, 0.01, 0.99)
        # Step 1: aggregate node features onto edges
        # Project node features
        for layer in range(self.layer_node2edge):
            if self.layer_node2edge == 1:
                h_projected = self.node_to_edge_projection[layer](h)
            elif layer == 0:
                h_projected = self.node_to_edge_projection[layer](h)
                h_projected = torch.nn.functional.relu(h_projected)
            elif layer == self.layer_node2edge - 1:
                h_projected = self.node_to_edge_projection[layer](h_projected)
            else:
                h_projected = self.node_to_edge_projection[layer](h_projected)
                h_projected = torch.nn.functional.relu(h_projected)
        h_projected = h_projected.view(h_projected.size(0), h_projected.size(1), 2) # batch*node*2*n

        # Aggregate node features onto edges
        node_aggregated_w = torch.matmul(self.adj_matrix, h_projected[:, :, 0].unsqueeze(dim=2))  # adj_matrix: edge*node mul batch * node * 1
        node_aggregated_u = torch.matmul(self.adj_matrix, h_projected[:, :, 1].unsqueeze(dim=2))
        node_aggregated = torch.cat((node_aggregated_w, node_aggregated_u), dim=2)

        # Update edge features
        e_updated = e + node_aggregated # shape: (3, 2, n)

        # Step 2: aggregate edge features onto nodes
        # Flatten each edge (2, n) feature to 1D (2*n,)
        e_flattened = e_updated.view(e_updated.size(0), e_updated.size(1), 2)  # shape: (3, 2*n)
        # Linear projection to 1D
        for layer in range(self.layer_edge2node):
            if self.layer_edge2node == 1:
                projected_edges = self.edge_to_node_projection[layer](e_flattened)
            elif layer == 0:
                projected_edges = self.edge_to_node_projection[layer](e_flattened)
                projected_edges = torch.nn.functional.relu(projected_edges)
            elif layer == self.layer_edge2node - 1:
                projected_edges = self.edge_to_node_projection[layer](projected_edges)
            else:
                projected_edges = self.edge_to_node_projection[layer](projected_edges)
                projected_edges = torch.nn.functional.relu(projected_edges)
        projected_edges = projected_edges.squeeze(1)  # shape: (3,)

        # Aggregate edge features onto nodes
        aggregated_edges = torch.matmul(self.adj_matrix.T, projected_edges)  # shape: (3,)

        # Update node features
        h_updated = h + aggregated_edges  # shape: (3,)
        # Step 3: update node features
        # Extended adjacency matrix: include neighboring nodes
        extended_node_matrix = compute_extended_adjacency(self.node_matrix, self.hops_node)
        # Compute node degree matrix
        D_node = torch.diag(torch.sum(extended_node_matrix, dim=1))
        # Compute normalized adjacency D^{-1/2} * A * D^{-1/2}
        D_inv_sqrt_node = torch.linalg.inv(torch.sqrt(D_node))
        node_matrix_hat = torch.matmul(torch.matmul(D_inv_sqrt_node, extended_node_matrix), D_inv_sqrt_node)
        # Convolve to update node features
        H_intermediate_node = torch.matmul(node_matrix_hat, h_updated)
        # Update node features
        for layer in range(self.layer_node):
            if self.layer_node == 1:
                h_updated = self.node_update[layer](H_intermediate_node).squeeze(dim=2)
            elif layer == 0:
                h_updated = self.node_update[layer](H_intermediate_node)
                h_updated = torch.nn.functional.tanh(h_updated)
            elif layer == self.layer_node - 1:
                h_updated = self.node_update[layer](h_updated).squeeze(dim=2)
            else:
                h_updated = self.node_update[layer](h_updated)
                h_updated = torch.nn.functional.tanh(h_updated)
        # Ensure non-negative water level
        h_updated = torch.nn.functional.relu(h_updated)

        # Step 4: update edge features
        # Extended adjacency matrix: include neighboring edges
        extended_edge_matrix = compute_adjacency_matrix(self.edge_matrix, self.hops_edge)
        # Compute edge degree matrix
        D_edge = torch.diag(torch.sum(extended_edge_matrix, dim=1))
        # Compute normalized adjacency D^{-1/2} * A * D^{-1/2}
        D_inv_sqrt_edge = torch.linalg.inv(torch.sqrt(D_edge))
        edge_matrix_hat = torch.matmul(torch.matmul(D_inv_sqrt_edge, extended_edge_matrix), D_inv_sqrt_edge)
        # Convolve to update edge features
        H_intermediate_edge = torch.matmul(edge_matrix_hat, e_updated.view(e_updated.size(0), e_updated.size(1), 2))
        # Update edge features
        for layer in range(self.layer_edge):
            if self.layer_edge == 1:
                e_updated = self.edge_update[layer](H_intermediate_edge).view(e_updated.size(0), e_updated.size(1), 2, 1)
            elif layer == 0:
                e_updated = self.edge_update[layer](H_intermediate_edge)
                e_updated = torch.nn.functional.relu(e_updated)
            elif layer == self.layer_edge - 1:
                e_updated = self.edge_update[layer](e_updated).view(e_updated.size(0), e_updated.size(1), 2, 1)
            else:
                e_updated = self.edge_update[layer](e_updated)
                e_updated = torch.nn.functional.relu(e_updated)
        # Clamp pipe occupancy to [0, 1]
        e_updated[:, :, 0, :] = torch.nn.functional.sigmoid(e_updated[:, :, 0, :])
        e_updated[:, :, 1, :] = e_updated[:, :, 1, :]
        e_updated = e_updated.squeeze(dim=3)
        return h_updated, e_updated, lambda1
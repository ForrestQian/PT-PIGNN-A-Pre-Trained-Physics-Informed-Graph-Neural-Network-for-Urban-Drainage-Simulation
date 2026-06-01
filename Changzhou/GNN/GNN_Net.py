import torch
import torch.nn as nn
from Function import Area
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

# GCN model
class GNN_Model(nn.Module):
    #  input_num, pre_lstm_node, pre_lstm_edge,
    def __init__(self, resolution, hops_node, hops_edge,
                 forward_node, forward_edge, length_index,
                 transcov_hidden, transcov_edge,
                 adj_matrix, node_matrix, edge_matrix):
        super(GNN_Model, self).__init__()
        self.length_index = length_index
        self.num_nodes = node_matrix.shape[0]
        self.num_edges = edge_matrix.shape[0]

        # Node update
        self.node_update = nn.ModuleList()
        if forward_node is not None:
            pre_neurons = 2 * 2
            for neurons in forward_node:
                self.node_update.append(nn.Linear(pre_neurons, neurons))
                pre_neurons = neurons
            self.node_update.append(nn.Linear(pre_neurons, 300))
        else:
            self.node_update.append(nn.Linear(4*1, 300))

        # Edge update
        self.edge_update_1 = nn.ModuleList()
        self.edge_update_2 = nn.ModuleList()
        if forward_edge is not None:
            pre_neurons = 2 * 2
            for neurons in forward_edge:
                self.edge_update_1.append(nn.Linear(pre_neurons, neurons))
                self.edge_update_2.append(nn.Linear(pre_neurons, neurons))
                pre_neurons = neurons
            self.edge_update_1.append(nn.Linear(pre_neurons, 300))
            self.edge_update_2.append(nn.Linear(pre_neurons, 300))
        else:
            self.edge_update_1 = nn.Linear(4*1, 300)
            self.edge_update_2 = nn.Linear(4*1, 300)

        self.lambda1 = torch.nn.Parameter(torch.tensor([0.50]))
        self.lambda2 = torch.nn.Parameter(torch.tensor([0.50]))
        self.lambda3 = torch.nn.Parameter(torch.tensor([0.50]))
        self.resolution = resolution
        self.hops_node = hops_node
        self.hops_edge = hops_edge
        self.adj_matrix = adj_matrix
        self.edge_matrix = edge_matrix
        self.node_matrix = node_matrix
        self.layer_node = len(self.node_update)
        self.layer_edge = len(self.edge_update_1)

    def forward(self, h, e, D, e_edge_matrix, e_node_matrix):
        # Clamp lambda to a reasonable range
        lambda1 = torch.clamp(self.lambda1, 0.01, 0.99)
        lambda2 = torch.clamp(self.lambda2, 0.01, 0.99)
        lambda3 = torch.clamp(self.lambda3, 0.01, 0.99)

        # Step 1: aggregate node features onto edges

        # Aggregate node features onto edges
        node_aggregated_h = torch.matmul(self.adj_matrix, h[:, :, 0].T).T.unsqueeze(dim=2) # adj_matrix: edge*node mul batch * node * n
        node_aggregated_q = torch.matmul(self.adj_matrix, h[:, :, 1].T).T.unsqueeze(dim=2)
        node_aggregated = torch.cat((node_aggregated_h, node_aggregated_q), dim=2)
        # Update edge features
        e_updated = torch.concatenate((e, node_aggregated), dim=2)

        # Step 2: aggregate edge features onto nodes

        # Aggregate edge features onto nodes
        edge_aggregated_w = torch.matmul(self.adj_matrix.T, e[:, :, 0].T).T.unsqueeze(dim=2)
        edge_aggregated_u = torch.matmul(self.adj_matrix.T, e[:, :, 1].T).T.unsqueeze(dim=2)
        edge_aggregated = torch.cat((edge_aggregated_w, edge_aggregated_u), dim=2)
        # Update node features
        h_updated = torch.concatenate((h, edge_aggregated), dim=2)

        # Step 3: update node features
        # Extended adjacency matrix: include neighboring nodes
        extended_node_matrix = compute_extended_adjacency(self.node_matrix, self.hops_node)
        # Compute node degree matrix
        D_node = torch.diag(torch.sum(extended_node_matrix, dim=1))
        # Compute normalized adjacency D^{-1/2} * A * D^{-1/2}
        D_inv_sqrt_node = torch.linalg.inv(torch.sqrt(D_node))
        node_matrix_hat = torch.matmul(torch.matmul(D_inv_sqrt_node, extended_node_matrix), D_inv_sqrt_node)
        # Convolve to update node features
        H_intermediate_node = torch.matmul(node_matrix_hat, h_updated).reshape((h_updated.shape[0], self.num_nodes, -1))
        # Update node features
        for layer in range(self.layer_node):
            if self.layer_node == 1:
                h_updated = self.node_update[layer](H_intermediate_node).squeeze(dim=2)
            elif layer == 0:
                h_updated = self.node_update[layer](H_intermediate_node)
                h_updated = torch.nn.functional.relu(h_updated)
            elif layer == self.layer_node - 1:
                h_updated = self.node_update[layer](h_updated).squeeze(dim=2)
            else:
                h_updated = self.node_update[layer](h_updated)
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
        H_intermediate_edge = torch.matmul(edge_matrix_hat, e_updated).reshape((e_updated.shape[0], self.num_edges, -1))
        # Update edge features
        for layer in range(self.layer_edge):
            if self.layer_edge == 1:
                flow_updated = self.edge_update_1[layer](H_intermediate_edge)
                velocity_updated = self.edge_update_2[layer](H_intermediate_edge)
            elif layer == 0:
                flow_updated = self.edge_update_1[layer](H_intermediate_edge)
                flow_updated = torch.nn.functional.relu(flow_updated)
                velocity_updated = self.edge_update_2[layer](H_intermediate_edge)
                velocity_updated = torch.nn.functional.relu(velocity_updated)
            elif layer == self.layer_edge - 1:
                flow_updated = self.edge_update_1[layer](flow_updated)
                velocity_updated = self.edge_update_2[layer](velocity_updated)
            else:
                flow_updated = self.edge_update_1[layer](flow_updated)
                flow_updated = torch.nn.functional.relu(flow_updated)
                velocity_updated = self.edge_update_2[layer](velocity_updated)
                velocity_updated = torch.nn.functional.relu(velocity_updated)

        flow_updated = flow_updated[:, :, -1].unsqueeze(dim=2)
        velocity_updated = velocity_updated[:, :, -1].unsqueeze(dim=2)
        edge_out = torch.concatenate((flow_updated, velocity_updated), dim=2)
        return h_updated[:, :, -1], edge_out, lambda1, lambda2, lambda3
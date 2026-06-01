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
class GCN_Model(nn.Module):
    #  input_num, pre_lstm_node, pre_lstm_edge,
    def __init__(self, resolution, hops_node, hops_edge,
                 forward_node, forward_edge, length_index,
                 transcov_hidden, transcov_edge,
                 adj_matrix, node_matrix, edge_matrix):
        super(GCN_Model, self).__init__()
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


        self.flow_full = nn.ModuleList()
        self.velocity_full = nn.ModuleList()
        # Edge CNN upsampling (per-meter average -> global average)
        self.edge_transconv = nn.ModuleList()
        if transcov_hidden is not None:
            pre_neurons = self.num_edges
            for neurons in transcov_hidden:
                self.flow_full.append(nn.Linear(pre_neurons, neurons))
                self.velocity_full.append(nn.Linear(pre_neurons, neurons))
                pre_neurons = neurons
                self.flow_full.append(nn.ReLU())
                self.velocity_full.append(nn.ReLU())
            self.flow_full.append(nn.Linear(pre_neurons, sum(self.length_index)))
            self.velocity_full.append(nn.Linear(pre_neurons, sum(self.length_index)))
        else:
            self.flow_full.append(nn.Linear(self.num_edges, sum(self.length_index)))
            self.velocity_full.append(nn.Linear(self.num_edges, sum(self.length_index)))

        # Add a transposed convolutional layer for upsampling
        in_channels = 2
        for i, out_channels in enumerate(transcov_edge):
            self.edge_transconv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),  # spatial kernel 3, temporal kernel 3
                    padding=(1, 1)  # preserve spatial and temporal dimensions
                )
            )
            self.edge_transconv.append(nn.ReLU())
            in_channels = out_channels
        self.edge_transconv.append(
            nn.Conv2d(
                in_channels=transcov_edge[-1],
                out_channels=2,
                kernel_size=(3, 3),  # spatial kernel 3, temporal kernel 3
                padding=(1, 1)  # preserve spatial and temporal dimensions
            )
        )

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

    def forward(self, h, e, D, D_full, e_edge_matrix, e_node_matrix):
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

        edge_updated = torch.concatenate((flow_updated.unsqueeze(dim=2), velocity_updated.unsqueeze(dim=2)), dim=2)
        flow_updated = flow_updated.permute(0, 2, 1)
        velocity_updated = velocity_updated.permute(0, 2, 1)
        origin_flow = e[:, :, 0]
        origin_velocity = e[:, :, 1]

        for layer in self.flow_full:
            flow_updated = layer(flow_updated)
            origin_flow = layer(origin_flow)
        for layer in self.velocity_full:
            velocity_updated = layer(velocity_updated)
            origin_velocity = layer(origin_velocity)

        flow_updated = flow_updated.permute(0, 2, 1)
        velocity_updated = velocity_updated.permute(0, 2, 1)

        flow_full = flow_updated
        velocity_full = velocity_updated

        # Step 5: edge CNN upsampling

        expanded_e = torch.concatenate((flow_full.unsqueeze(dim=1), velocity_full.unsqueeze(dim=1)), dim=1)

        for layer in self.edge_transconv:
            expanded_e = layer(expanded_e)
        expanded_e = expanded_e.permute(0, 2, 3, 1)
        expanded_e[:, :, :, 0] = torch.nn.functional.sigmoid(expanded_e[:, :, :, 0]) * 0.98 + 0.01
        # expanded_e[:, :, :, 1] = torch.nn.functional.relu(expanded_e[:, :, :, 1])
        h_out = torch.zeros_like(h[:, :, 0])
        edge_out = torch.zeros_like(e)

        curr_index = 0
        for i in range(self.num_nodes):
            if i == 0:
                h_out[:, i] = expanded_e[:, curr_index, -1, 0] * D[i]
                curr_index = curr_index + self.length_index[i]
            elif i == self.num_nodes - 1:
                h_out[:, i] = expanded_e[:, curr_index - 1, -1, 0] * D[i - 1]
            else:
                h_out[:, i] = (expanded_e[:, curr_index, -1, 0] * D[i] + expanded_e[:, curr_index - 1, -1, 0] * D[
                    i - 1]) / 2
                curr_index = curr_index + self.length_index[i]

        for i in range(self.num_edges):
            point = sum(self.length_index[:i])
            end = sum(self.length_index[:i]) + self.length_index[i] - 1
            edge_out[:, i, 0] = torch.mean(expanded_e[:, point:end, -1, 0], dim=1)
            edge_out[:, i, 1] = torch.mean(expanded_e[:, point:end, -1, 1], dim=1)

        return h_out, edge_out, expanded_e, lambda1, lambda2, lambda3
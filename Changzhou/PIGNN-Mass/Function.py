import torch
import numpy as np
import random

def create_adjacency_matrices(edge_index, num_nodes, num_edges, device):
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
    return adj_matrix, node_matrix, edge_matrix

def multiply_list_with_tensor(input_list, tensor_2d):
    vec = torch.tensor(input_list, dtype=tensor_2d.dtype, device=tensor_2d.device)

    # Ensure vec is 2D (1, 162)
    vec = vec.unsqueeze(0) if vec.dim() == 1 else vec

    # Matrix multiply: (1, 162) @ (162, 9775) -> (1, 9775)
    result = torch.matmul(vec, tensor_2d)

    # Transpose via transpose(0, 1) instead of .T
    result = result.transpose(0, 1)  # (9775, 1)

    return result

def weight_init(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def Area(w, D):
    return D ** 2 / 4 * torch.arccos(1 - 2 * w) - D ** 2 / 2 * (1 - 2 * w) * torch.sqrt(w - w ** 2)

def DX_Get(x):
    forward_diff = x[:, 1, :] - x[:, 0, :]
    forward_diff = forward_diff.unsqueeze(dim=1)
    center_diff = (x[:, 2:, :] - x[:, :-2, :]) / 2
    backward_diff = x[:, -1, :] - x[:, -2, :]
    backward_diff = backward_diff.unsqueeze(dim=1)

    result = torch.cat([forward_diff, center_diff, backward_diff], dim=1)
    # Assert output tensor shape matches input
    assert result.size() == x.size()
    return result

def DT_Get(x):
    forward_diff = x[:, :, 1] - x[:, :, 0]
    forward_diff = forward_diff.unsqueeze(dim=2)
    center_diff = (x[:, :, 2:] - x[:, :, :-2]) / 2
    backward_diff = x[:, :, -1] - x[:, :, -2]
    backward_diff = backward_diff.unsqueeze(dim=2)

    result = torch.cat([forward_diff, center_diff, backward_diff], dim=2)
    # Assert output tensor shape matches input
    assert result.size() == x.size()
    return result

def PDE_Loss(q, expanded_e, n_index, D_index, g, S0_index, length_index):
    q = q[:, :, :]
    w = expanded_e[:, :, :, 0]
    u = expanded_e[:, :, :, 1]

    s = Area(w, D_index)
    y = w * D_index
    flow = s * u
    dAdt = DT_Get(s)
    dQdx = DX_Get(flow)
    dudt = DT_Get(u)
    dudx = DX_Get(u)
    dydx = DX_Get(y)
    theta = torch.arccos(1 - w * 2) * 2

    R = s / D_index / theta

    for i in range(len(length_index)):
        dQdx[:, sum(length_index[:i]), :] = dQdx[:, sum(length_index[:i]), :] - q[:, i, :]

    PDE_1 = dAdt + dQdx
    PDE_2 = dudt / g + u * dudx / g + dydx + n_index ** 2 * u * torch.abs(u) * R ** (-4 / 3) - S0_index
    for i in range(len(length_index)):
        PDE_2[:, sum(length_index[:i]), :] = 0

    return PDE_1, PDE_2

def nse_loss(y_true, y_pred):
    # Compute mean squared error (MSE)
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    # Compute variance of observations
    variance = torch.mean((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    # Compute NSE
    nse = 1 - mse / variance
    return nse

import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from sympy.physics.control import backward_diff


def nse_loss(y_true, y_pred):
    # Compute mean squared error (MSE)
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    # Compute variance of observations
    variance = torch.mean((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    # Compute NSE
    nse = 1 - mse / variance
    return nse

def Area(w, D):
    return D ** 2 / 4 * torch.arccos(1 - 2 * w) - D ** 2 / 2 * (1 - 2 * w) * torch.sqrt(w - w ** 2)

def Area_np(w, D):
    return D ** 2 / 4 * np.arccos(1 - 2 * w) - D ** 2 / 2 * (1 - 2 * w) * np.sqrt(w - w ** 2)

def inverse_f(y, D):
    alpha = 4 * y / (torch.pi * D ** 2)
    theta = (0.031715 - 12.79384 * alpha + 8.28479 * torch.sqrt(alpha))
    delta_theta = (2 * torch.pi * alpha - (theta - torch.sin(theta))) / (1 - torch.cos(theta) + 1)
    while not (abs(delta_theta) <= 0.1).all():
        delta_theta = (2 * torch.pi * alpha - (theta - torch.sin(theta))) / (1 - torch.cos(theta) + 1)
        theta = theta + delta_theta
    return theta

def Graph_plot(edge_index:torch.tensor, node_index:torch.tensor):
    G = nx.DiGraph()
    node_plot = node_index.clone().numpy().tolist()
    node_plot = [str(element) for element in node_plot]
    edge_index_plot = edge_index.clone().t().contiguous().numpy().tolist()
    edge_index_plot = [[str(element) for element in sublist] for sublist in edge_index_plot]
    G.add_nodes_from(node_plot)
    G.add_edges_from(edge_index_plot)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()
    return G

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
    q = q[:, :, :] * 100 / 3600
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# Weight initialization
def weight_init(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    return model

# Subset generation
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path)
        for i in range(start, len(nums)):
            backtrack(i + 1, path + [nums[i]])
    backtrack(0, [])
    return result
# Increasing subset generation
def subsets_increasing(nums):
    result = []
    def backtrack(start, path):
        result.append(path)
        for i in range(start, len(nums)):
            backtrack(i + 1, path + [nums[i]])
    nums.sort()
    backtrack(0, [])
    return result
# Decreasing subset generation
def subsets_decreasing(nums):
    result = []
    def backtrack(start, path):
        result.append(path)
        for i in range(start, len(nums)):
            backtrack(i + 1, path + [nums[i]])
    nums.sort(reverse=True)  # Sort descending
    backtrack(0, [])
    return result

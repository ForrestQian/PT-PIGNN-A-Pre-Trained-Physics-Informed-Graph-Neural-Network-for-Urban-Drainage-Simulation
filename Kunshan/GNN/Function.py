import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
def nse_loss(y_true, y_pred):
    # Compute mean squared error (MSE)
    mse = torch.mean((y_true - y_pred) ** 2)
    # Compute variance of observations
    variance = torch.mean((y_true - torch.mean(y_true)) ** 2)
    # Compute NSE
    nse = 1 - mse / variance
    return nse

def Area(w, D):
    return D ** 2 / 4 * torch.arccos(1 - 2 * w) - D ** 2 / 2 * (1 - 2 * w) * torch.sqrt(w - w ** 2)

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

def PDE_Loss(node_feature, edge_feature, node_updated, edge_updated, n_index, D_index, g, S0_index, edge_index):
    h_0 = node_feature[:, :, 0]
    q = node_feature[:, :, 1]
    h_t = node_updated
    u_0 = edge_feature[:, :, 1, :]
    u_t = edge_updated[:, :, 1, :]
    w_0 = edge_feature[:, :, 0, :]
    w_t = edge_updated[:, :, 0, :]

    s_t = Area(w_t, D_index)
    s_0 = Area(w_0, D_index)
    dwdt = w_t - w_0
    dsdt = s_t - s_0
    dudt = u_t - u_0
    dwdx = torch.zeros_like(w_t)
    dsdx = torch.zeros_like(s_t)
    dudx = torch.zeros_like(u_t)

    for m in range(1, u_t.shape[1] - 1):
        dwdx[:, :, m] = (w_t[:, :, m+1] - w_t[:, :, m-1]) / 2
        dsdx[:, :, m] = (s_t[:, :, m+1] - s_t[:, :, m-1]) / 2
        dudx[:, :, m] = (u_t[:, :, m+1] - u_t[:, :, m-1]) / 2
    # TODO: boundary dsdx, dudx from water level BC
    for m_list in range(dwdx.size(1)):
        dwdx[:, m_list, 0] = (w_t[:, m_list, 1] - torch.maximum(torch.minimum(h_t[:, edge_index[0, m_list]] / D_index[:, m_list, 0], torch.ones_like(h_t[:, edge_index[0, m_list]])), torch.zeros_like(h_t[:, edge_index[0, m_list]]))) / 2
        dwdx[:, m_list, -1] = (torch.maximum(torch.minimum(h_t[:, edge_index[1, m_list]] / D_index[:, m_list, 0], torch.ones_like(h_t[:, edge_index[1, m_list]])), torch.zeros_like(h_t[:, edge_index[1, m_list]]))- w_t[:, m_list, -2]) / 2

        dsdx[:, m_list, 0] = (s_t[:, m_list, 1] - Area(torch.maximum(torch.minimum(h_t[:, edge_index[1, m_list]] / D_index[:, m_list, 0], torch.ones_like(h_t[:, edge_index[1, m_list]])), torch.zeros_like(h_t[:, edge_index[1, m_list]])), D_index[0, m_list, 0])) / 2
        dsdx[:, m_list, -1] = (Area(torch.maximum(torch.minimum(h_t[:, edge_index[1, m_list]] / D_index[:, m_list, 0], torch.ones_like(h_t[:, edge_index[1, m_list]])), torch.zeros_like(h_t[:, edge_index[1, m_list]])), D_index[0, m_list, 0])  - s_t[:, m_list, -2]) / 2
        dudx[:, m_list, 0] = (u_t[:, m_list, 1] - 0) / 2
        dudx[:, m_list, -1] = (0 - u_t[:, m_list, -2]) / 2

    # Hydrodynamic equations
    PDE_Loss_1 = dsdt - dudx * s_t - dsdx * u_t

    for m_edge in range(PDE_Loss_1.size(1)):
        node_id = edge_index[0, m_edge]
        PDE_Loss_1[:, m_edge, 0] = PDE_Loss_1[:, m_edge, 0] - q[:, node_id]
    PDE_Loss_2 = dudt + u_t * dudx + g * D_index * dwdx + g * (n_index ** 2 * u_t ** 2 * (s_t / D_index / torch.arccos(1 - 2 * w_t) ** (-4 / 3)) - S0_index)
    return PDE_Loss_1, PDE_Loss_2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# Weight initialization
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
     # Whether layer is batch normalization
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

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

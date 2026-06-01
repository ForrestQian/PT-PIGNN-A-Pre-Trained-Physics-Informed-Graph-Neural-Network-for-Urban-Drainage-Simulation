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

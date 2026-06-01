from PIGCN_Train import PIGCN_train
import torch
from PIGCN_Net import GCN_Dataset, GCN_Model
from torch.utils.data import DataLoader
from hyperopt import fmin, tpe, hp, Trials
import pickle
from Function import subsets, subsets_increasing, subsets_decreasing

# lr, weight_decay -- optimizer hyperparameters
# lr: 1e-5, 1e-4, 1e-3
# weight_decay: 1e-7, 1e-6, 1e-5
# hidden_kernels_* and hop params -- network hyperparameters
# hidden_kernels_node: any combination from [8, 16, 32]
# hidden_kernels_edge: any combination from [32, 64, 128]
# hidden_kernels_node2edge: increasing combo from [32, 64, 128]
# hidden_kernels_edge2node: decreasing combo from [128, 64, 32]
# edge_hop: 2, 3, 4
# node_hop: 2, 3, 4

TRIALS_PATH = ''  # Set locally before running

# region Parameter import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 93
dev_size = 962
epochs = 1
train_node_index = [2, 5, 8, 11, 14, 17]
train_edge_index = [5, 14]
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
train_dataset = GCN_Dataset(edge_index, length_index, file_path='')
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataset = GCN_Dataset(edge_index, length_index, file_path='')
devloader = DataLoader(test_dataset, batch_size=dev_size, shuffle=False, drop_last=True)
# endregion
# Define objective function
def objective(params):
    lr = params['lr']
    weight_decay = params['weight_decay']
    hidden_kernels_node = params['hidden_kernels_node']
    hidden_kernels_edge = params['hidden_kernels_edge']
    hidden_kernels_node2edge = params['hidden_kernels_node2edge']
    hidden_kernels_edge2node = params['hidden_kernels_edge2node']
    hops_node = params['edge_hop']
    hops_edge = params['node_hop']
    # Log parameters
    print(params)
    model = GCN_Model(n=n, hops_node=hops_node, hidden_kernels_node=hidden_kernels_node,
                      hidden_kernels_node2edge=hidden_kernels_node2edge,
                      hops_edge=hops_edge, hidden_kernels_edge=hidden_kernels_edge,
                      hidden_kernels_edge2node=hidden_kernels_edge2node,
                      adj_matrix=adj_matrix, node_index=node_index, node_matrix=node_matrix,
                      edge_matrix=edge_matrix, edge_index=edge_index).to(device)
    # Train model
    mean_nse, lowerest_nse = PIGCN_train(model=model, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                                         dev_size=dev_size,
                                         D_input=D_input, n_input=n_input, S0_input=S0_input,
                                         epochs=epochs, train_node_index=train_node_index,
                                         train_edge_index=train_edge_index,
                                         length_index=length_index, device=device,
                                         edge_index=edge_index, dataloader=dataloader, devloader=devloader)

    # Hyperopt minimizes, so return 1 minus accuracy
    return -lowerest_nse

space_node = subsets([8, 16, 32])
space_edge = subsets([32, 64, 128])
space_node2edge = subsets_increasing([32, 64, 128])
space_edge2node = subsets_decreasing([32, 64, 128])
space_lr = [1e-5, 1e-4, 1e-3]
space_weight_decay = [1e-7, 1e-6, 1e-5]
space_edge_hop = [2, 3, 4]
space_node_hop = [2, 3, 4]

# Hyperopt search space
space = {
    'lr': hp.choice('lr', space_lr),
    'weight_decay': hp.choice('weight_decay', space_weight_decay),
    'hidden_kernels_node': hp.choice('hidden_kernels_node', space_node),
    'hidden_kernels_edge': hp.choice('hidden_kernels_edge', space_edge),
    'hidden_kernels_node2edge': hp.choice('hidden_kernels_node2edge', space_node2edge),
    'hidden_kernels_edge2node': hp.choice('hidden_kernels_edge2node', space_edge2node),
    'edge_hop': hp.choice('edge_hop', space_edge_hop),
    'node_hop': hp.choice('node_hop', space_node_hop)
}

# Hyperparameter optimization
# Try loading Trials from file
try:
    with open(TRIALS_PATH, 'rb') as f:
        trials = pickle.load(f)
        print("Loaded previous trials.")
        # Print saved results
        print(f"Loaded Trials: {trials.trials}")
except FileNotFoundError:
    trials = Trials()
    print("No previous trials found; starting new search.")
# Run hyperparameter optimization
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
# Save trials object
with open(TRIALS_PATH, 'wb') as f:
    pickle.dump(trials, f)







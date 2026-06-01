import torch
from PIGCN_Net import GCN_Dataset, GCN_Model
from torch.utils.data import DataLoader
from visdom import Visdom
from Function import setup_seed, weight_init, PDE_Loss, nse_loss, subsets, subsets_increasing, subsets_decreasing
import pandas as pd
import numpy as np
from visdom import Visdom
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
weight_decay = space_weight_decay[2]
hidden_kernels_node = space_node[3]
hidden_kernels_edge = space_edge[0]
hidden_kernels_node2edge = space_node2edge[2]
hidden_kernels_edge2node = space_edge2node[2]
hops_edge = space_edge_hop[1]
hops_node = space_node_hop[1]
print('Best_params:')
print('lr:', lr)
print('weight_decay:', weight_decay)
print('hidden_kernels_node:', hidden_kernels_node)
print('hidden_kernels_edge:', hidden_kernels_edge)
print('hidden_kernels_node2edge:', hidden_kernels_node2edge)
print('hidden_kernels_edge2node:', hidden_kernels_edge2node)

# endregion

# region Parameter import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 93
dev_size = 962
epochs = 1000
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

def PIGCN_train(length_index, dev_size, train_node_index, train_edge_index,                          ## training nodes
                D_input, n_input, S0_input,                                                    ## physical properties
                epochs, model, batch_size, device, lr, weight_decay,                                 ## training hyperparameters
                edge_index, dataloader, devloader):
    model = model.to(device)
    viz = Visdom(env='1130_Train_bestPIGCN')
    draw_window1 = 'TotalLoss'
    draw_window2 = 'Data_1Loss'
    draw_window3 = 'Data_2Loss'
    draw_window4 = 'PDE_1Loss'
    draw_window5 = 'PDE_2Loss'
    time = 0
    ## data
    setup_seed(14)
    lowerest_nse = -np.inf
    mean_nse = -np.inf
    nse_best_epoch = 0
    nse_min_best_epoch = 0
    max_length = torch.max(length_index).to(device)
    num_edges = len(length_index)
    D_feature = torch.zeros((num_edges, max_length))
    D_feature_dev = torch.zeros((num_edges, max_length))
    D_feature = D_feature + D_input.unsqueeze(dim=1)
    D_feature_dev = D_feature_dev + D_input.unsqueeze(dim=1)
    D_feature = D_feature.unsqueeze(0).repeat(batch_size, 1, 1)
    D_feature_dev = D_feature_dev.unsqueeze(0).repeat(dev_size, 1, 1)
    D_feature = D_feature.to(device)
    D_feature_dev = D_feature_dev.to(device)

    S0_feature = torch.zeros((num_edges, max_length))
    S0_feature = S0_feature + S0_input.unsqueeze(dim=1)
    S0_feature = S0_feature.unsqueeze(0).repeat(batch_size, 1, 1)
    S0_feature = S0_feature.to(device)

    n_feature = torch.zeros((num_edges, max_length))
    n_feature = n_feature + n_input.unsqueeze(dim=1)
    n_feature = n_feature.unsqueeze(0).repeat(batch_size, 1, 1)
    n_feature = n_feature.to(device)
    g = 9.8

    model = model.apply(weight_init)
    # Reset loss bias weights
    model.lambda1 = torch.nn.Parameter(torch.tensor([0.50]))
    model.lambda2 = torch.nn.Parameter(torch.tensor([0.50]))
    model.lambda3 = torch.nn.Parameter(torch.tensor([0.50]))
    model = model.to(device)

    PDE_Label = torch.zeros((batch_size, num_edges, max_length)).to(device)
    Criterion = torch.nn.MSELoss(reduction='mean').to(device)
    Optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, amsgrad=False)

    torch.save(model.state_dict(), 'PIGCN_epoch_'+str(0)+'.pth')

    for i in range(epochs):
        for index, data in enumerate(devloader, 0):
            model.eval()
            node_feature, edge_feature, output_node_feature, output_edge_feature = data

            # Select training data by train_node_index and train_edge_index
            train_output_node_feature = torch.zeros((dev_size, len(train_node_index)))
            for l in range(len(train_node_index)):
                train_output_node_feature[:, l] = output_node_feature[:, train_node_index[l] - 1]
            output_node_feature = train_output_node_feature
            train_output_edge_feature = torch.zeros((dev_size, len(train_edge_index), 2))
            for l in range(len(train_edge_index)):
                train_output_edge_feature[:, l, :] = output_edge_feature[:, train_edge_index[l] - 1, :]
            output_edge_feature = train_output_edge_feature

            node_feature = node_feature.to(device)
            edge_feature = edge_feature.to(device)
            output_node_feature = output_node_feature.to(device)
            output_edge_feature = output_edge_feature.to(device)
            node_updated, q_updated, edge_updated, lambda1, lambda2, lambda3 = model(node_feature, edge_feature, D_feature_dev)

            train_node_updated = torch.zeros((dev_size, len(train_node_index))).to(device)
            for l in range(len(train_node_index)):
                train_node_updated[:, l] = node_updated[:, train_node_index[l] - 1]
            node_updated = train_node_updated.to(device)

            train_edge_updated = torch.zeros((dev_size, len(train_edge_index), 2, max_length)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_updated[:, l, :] = edge_updated[:, train_edge_index[l] - 1, :, :]
            edge_updated = train_edge_updated.to(device)

            train_node_feature = torch.zeros((dev_size, len(train_node_index), 2)).to(device)
            for l in range(len(train_node_index)):
                train_node_feature[:, l, :] = node_feature[:, train_node_index[l] - 1, :]
            train_edge_feature = torch.zeros((dev_size, len(train_edge_index), 2, max_length)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_feature[:, l, :] = edge_feature[:, train_edge_index[l] - 1, :, :]
            edge_updated_w = edge_updated[:, :, 0, :].to(device)
            edge_updated_u = edge_updated[:, :, 1, :].to(device)

            mean_update_w = torch.tensor(
                [[torch.mean(edge_updated_w[m, k, :length_index[k]]) for k in range(len(train_edge_index))] for m in
                 range(edge_updated.size(0))]).to(device)
            mean_update_u = torch.tensor(
                [[torch.mean(edge_updated_u[m, k, :length_index[k]]) for k in range(len(train_edge_index))] for m in
                 range(edge_updated.size(0))]).to(device)
            nse_node = nse_loss(node_updated, output_node_feature).item()
            nse_edge_capacity = nse_loss(mean_update_w, output_edge_feature[:, :, 0]).item()
            nse_edge_velocity = nse_loss(mean_update_u, output_edge_feature[:, :, 1]).item()
            nse = (nse_node + nse_edge_capacity + nse_edge_velocity) / 3
            min_nse = min(nse_node, nse_edge_capacity, nse_edge_velocity)
            if nse > mean_nse:
                mean_nse = nse
                nse_best_epoch = i
            if min_nse > lowerest_nse:
                lowerest_nse = min_nse
                nse_min_best_epoch = i
            lambda1_copy = lambda1.cpu().detach().numpy()
            lambda2_copy = lambda2.cpu().detach().numpy()
            lambda3_copy = lambda3.cpu().detach().numpy()

            viz.line([lambda1_copy * lambda3_copy], [i], win='lambda', name='pde_1', update='append')
            viz.line([(1 - lambda1_copy) * lambda3_copy], [i], win='lambda', name='pde_2', update='append')
            viz.line([lambda2_copy * (1 - lambda3_copy)], [i], win='lambda', name='data_1', update='append')
            viz.line([(1 - lambda2_copy) * (1 - lambda3_copy)], [i], win='lambda', name='data_2', update='append')
            if i == 0:
                viz.line([nse_loss(y_pred=node_updated, y_true=output_node_feature).cpu().detach().numpy()], [0], win='NODE_NSE', opts=dict(title='NODE_NSE'))
                viz.line([nse_loss(y_pred=mean_update_w, y_true=output_edge_feature[:, :, 0]).cpu().detach().numpy()], [0], win='EDGE_w_NSE', opts=dict(title='EDGE_w'))
                viz.line([nse_loss(y_pred=mean_update_u, y_true=output_edge_feature[:, :, 1]).cpu().detach().numpy()], [0], win='EDGE_u_NSE', opts=dict(title='EDGE_u'))
                viz.line(node_updated[:, 1].cpu().detach().numpy(), [time for time in range(dev_size)], win='NODE_5',
                         name='Pred', update='append', opts=dict(title='NODE_5'))
                viz.line(output_node_feature[:, 1].cpu().detach().numpy(), [time for time in range(dev_size)], win='NODE_5',
                         name='True', update='append', opts=dict(title='NODE_5'))
                viz.line(mean_update_w[:, 0].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Capacity',
                         name='Pred', update='append', opts=dict(title='EDGE_5_6_Capacity'))
                viz.line(output_edge_feature[:, 0, 0].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Capacity',
                         name='True', update='append', opts=dict(title='EDGE_5_6_Capacity'))
                viz.line(mean_update_u[:, 0].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Velocity',
                         name='Pred', update='append', opts=dict(title='EDGE_5_6_Velocity'))
                viz.line(output_edge_feature[:, 0, 1].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Velocity',
                         name='True', update='append', opts=dict(title='EDGE_5_6_Velocity'))
            else:
                viz.line([nse_loss(y_pred=node_updated, y_true=output_node_feature).cpu().detach().numpy()], [i], win='NODE_NSE', update='append')
                viz.line([nse_loss(y_pred=mean_update_w, y_true=output_edge_feature[:, :, 0]).cpu().detach().numpy()], [i], win='EDGE_w_NSE', update='append')
                viz.line([nse_loss(y_pred=mean_update_u, y_true=output_edge_feature[:, :, 1]).cpu().detach().numpy()], [i], win='EDGE_u_NSE', update='append')
                viz.line(node_updated[:, 1].cpu().detach().numpy(), [time for time in range(dev_size)], win='NODE_5',
                         name='Pred', update='replace', opts=dict(title='NODE_5'))
                viz.line(output_node_feature[:, 1].cpu().detach().numpy(), [time for time in range(dev_size)], win='NODE_5',
                         name='True', update='replace', opts=dict(title='NODE_5'))
                viz.line(mean_update_w[:, 0].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Capacity',
                         name='Pred', update='replace', opts=dict(title='EDGE_5_6_Capacity'))
                viz.line(output_edge_feature[:, 0, 0].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Capacity',
                         name='True', update='replace', opts=dict(title='EDGE_5_6_Capacity'))
                viz.line(mean_update_u[:, 0].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Velocity',
                         name='Pred', update='replace', opts=dict(title='EDGE_5_6_Velocity'))
                viz.line(output_edge_feature[:, 0, 1].cpu().detach().numpy(), [time for time in range(dev_size)], win='EDGE_5_6_Velocity',
                         name='True', update='replace', opts=dict(title='EDGE_5_6_Velocity'))

        for index, data in enumerate(dataloader, 0):
            model.train()
            node_feature, edge_feature, output_node_feature, output_edge_feature = data
            # Select training data by train_node_index and train_edge_index
            train_output_node_feature = torch.zeros((batch_size, len(train_node_index)))
            for l in range(len(train_node_index)):
                train_output_node_feature[:, l] = output_node_feature[:, train_node_index[l]-1]
            output_node_feature = train_output_node_feature
            train_output_edge_feature = torch.zeros((batch_size, len(train_edge_index), 2))
            for l in range(len(train_edge_index)):
                train_output_edge_feature[:, l, :] = output_edge_feature[:, train_edge_index[l]-1, :]
            output_edge_feature = train_output_edge_feature

            node_feature = node_feature.to(device)
            edge_feature = edge_feature.to(device)
            output_node_feature = output_node_feature.to(device)
            output_edge_feature = output_edge_feature.to(device)
            node_updated, q_updated, edge_updated, lambda1, lambda2, lambda3 = model(node_feature, edge_feature, D_feature)
            PDE_1, PDE_2 = PDE_Loss(node_feature, edge_feature, node_updated, edge_updated, n_feature, D_feature, g,
                                    S0_feature, edge_index)
            train_node_updated = torch.zeros((batch_size, len(train_node_index))).to(device)
            for l in range(len(train_node_index)):
                train_node_updated[:, l] = node_updated[:, train_node_index[l]-1]
            node_updated = train_node_updated

            train_edge_updated = torch.zeros((batch_size, len(train_edge_index), 2, max_length)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_updated[:, l, :] = edge_updated[:, train_edge_index[l]-1, :, :]
            edge_updated = train_edge_updated

            train_node_feature = torch.zeros((batch_size, len(train_node_index), 2)).to(device)
            for l in range(len(train_node_index)):
                train_node_feature[:, l, :] = node_feature[:, train_node_index[l]-1, :]

            train_edge_feature = torch.zeros((batch_size, len(train_edge_index), 2, max_length)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_feature[:, l, :] = edge_feature[:, train_edge_index[l]-1, :, :]

            edge_updated_w = edge_updated[:, :, 0, :]
            edge_updated_u = edge_updated[:, :, 1, :]

            mean_update_w = torch.tensor([[torch.mean(edge_updated_w[m, k, :length_index[k]]) for k in range(len(train_edge_index))] for m in range(edge_updated.size(0))])
            mean_update_u = torch.tensor([[torch.mean(edge_updated_u[m, k, :length_index[k]]) for k in range(len(train_edge_index))] for m in range(edge_updated.size(0))])
            mean_edge_update = torch.cat((mean_update_w.unsqueeze(dim=2), mean_update_u.unsqueeze(dim=2)), dim=2)

            mean_edge_update = mean_edge_update.to(device)

            PDE_1 = Criterion(PDE_1, PDE_Label)
            PDE_2 = Criterion(PDE_2, PDE_Label)
            Data_1 = Criterion(mean_edge_update, output_edge_feature)
            Data_2 = Criterion(node_updated, output_node_feature)
            Loss = (PDE_1 * lambda1 + PDE_2 * (1 - lambda1)) * lambda3 + (Data_1 * lambda2 + Data_2 * (1 - lambda2)) * (1 - lambda3)
            # Backpropagate the loss
            Loss.backward()
            # Update model parameters
            Optimizer.step()
            # Clear gradients for next step
            Optimizer.zero_grad()
            loss1_copy = Loss.cpu().detach().numpy()
            loss2_copy = Data_1.cpu().detach().numpy()
            loss3_copy = Data_2.cpu().detach().numpy()
            loss4_copy = PDE_1.cpu().detach().numpy()
            loss5_copy = PDE_2.cpu().detach().numpy()
            if time == 0:
                viz.line([loss1_copy], [0], win=draw_window1, opts=dict(title=draw_window1))
                viz.line([loss2_copy], [0], win=draw_window2, opts=dict(title=draw_window2))
                viz.line([loss3_copy], [0], win=draw_window3, opts=dict(title=draw_window3))
                viz.line([loss4_copy], [0], win=draw_window4, opts=dict(title=draw_window4))
                viz.line([loss5_copy], [0], win=draw_window5, opts=dict(title=draw_window5))
                time += 1
            elif index % 10 == 0:
                viz.line([loss1_copy], [time], win=draw_window1, update='append')
                viz.line([loss2_copy], [time], win=draw_window2, update='append')
                viz.line([loss3_copy], [time], win=draw_window3, update='append')
                viz.line([loss4_copy], [time], win=draw_window4, update='append')
                viz.line([loss5_copy], [time], win=draw_window5, update='append')
                time += 1
            torch.save(model.state_dict(), 'PIGCN_epoch_' + str(i+1) + '.pth')
    print(nse_best_epoch, nse_min_best_epoch)
    return mean_nse, lowerest_nse

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
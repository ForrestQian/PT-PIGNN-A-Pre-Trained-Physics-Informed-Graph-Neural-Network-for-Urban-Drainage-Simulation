import torch
from Function import setup_seed, weight_init, nse_loss
import numpy as np
def GNN_train(dev_size, train_node_index, train_edge_index,                          ## training nodes
              epochs, model, batch_size, device, lr, weight_decay,                                 ## training hyperparameters
              dataloader, devloader):                                                  ## data
    setup_seed(14)
    lowerest_nse = -np.inf
    mean_nse = -np.inf

    model = model.apply(weight_init)
    # Reset loss bias weights
    model.lambda1 = torch.nn.Parameter(torch.tensor([0.50]))
    model = model.to(device)

    Criterion = torch.nn.MSELoss(reduction='mean').to(device)
    Optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, amsgrad=False)

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
            node_updated, edge_updated, lambda1 = model(node_feature, edge_feature)

            train_node_updated = torch.zeros((dev_size, len(train_node_index))).to(device)
            for l in range(len(train_node_index)):
                train_node_updated[:, l] = node_updated[:, train_node_index[l] - 1]
            node_updated = train_node_updated.to(device)

            train_edge_updated = torch.zeros((dev_size, len(train_edge_index), 2)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_updated[:, l, :] = edge_updated[:, train_edge_index[l] - 1, :]
            edge_updated = train_edge_updated.to(device)

            train_node_feature = torch.zeros((dev_size, len(train_node_index), 2)).to(device)
            for l in range(len(train_node_index)):
                train_node_feature[:, l, :] = node_feature[:, train_node_index[l] - 1, :]
            train_edge_feature = torch.zeros((dev_size, len(train_edge_index), 2)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_feature[:, l, :] = edge_feature[:, train_edge_index[l] - 1, :]
            edge_updated_w = edge_updated[:, :, 0].to(device)
            edge_updated_u = edge_updated[:, :, 1].to(device)

            mean_update_w = edge_updated_w
            mean_update_u = edge_updated_u
            nse_node = nse_loss(node_updated, output_node_feature).item()
            nse_edge_capacity = nse_loss(mean_update_w, output_edge_feature[:, :, 0]).item()
            nse_edge_velocity = nse_loss(mean_update_u, output_edge_feature[:, :, 1]).item()
            nse = (nse_node + nse_edge_capacity + nse_edge_velocity) / 3
            min_nse = min(nse_node, nse_edge_capacity, nse_edge_velocity)
            if nse > mean_nse:
                mean_nse = nse
            if min_nse > lowerest_nse:
                lowerest_nse = min_nse

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
            node_updated, edge_updated, lambda1 = model(node_feature, edge_feature)

            train_node_updated = torch.zeros((batch_size, len(train_node_index))).to(device)
            for l in range(len(train_node_index)):
                train_node_updated[:, l] = node_updated[:, train_node_index[l]-1]
            node_updated = train_node_updated

            train_edge_updated = torch.zeros((batch_size, len(train_edge_index), 2)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_updated[:, l, :] = edge_updated[:, train_edge_index[l]-1, :]
            edge_updated = train_edge_updated

            train_node_feature = torch.zeros((batch_size, len(train_node_index), 2)).to(device)
            for l in range(len(train_node_index)):
                train_node_feature[:, l, :] = node_feature[:, train_node_index[l]-1, :]

            train_edge_feature = torch.zeros((batch_size, len(train_edge_index), 2)).to(device)
            for l in range(len(train_edge_index)):
                train_edge_feature[:, l, :] = edge_feature[:, train_edge_index[l]-1, :]

            edge_updated_w = edge_updated[:, :, 0]
            edge_updated_u = edge_updated[:, :, 1]

            mean_update_w = edge_updated_w
            mean_update_u = edge_updated_u
            mean_edge_update = torch.cat((mean_update_w.unsqueeze(dim=2), mean_update_u.unsqueeze(dim=2)), dim=2)

            mean_edge_update = mean_edge_update.to(device)
            Data_1 = Criterion(mean_edge_update, output_edge_feature)
            Data_2 = Criterion(node_updated, output_node_feature)
            Loss = Data_1 * lambda1 + Data_2 * (1 - lambda1)
            # Backpropagate the loss
            Loss.backward()
            # Update model parameters
            Optimizer.step()
            # Clear gradients for next step
            Optimizer.zero_grad()
    return mean_nse, lowerest_nse



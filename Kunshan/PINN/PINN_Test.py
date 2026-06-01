import numpy as np
from Data_ReTransfer import data_retransfer
from Function import nse_loss
from torch.utils.data import DataLoader
import torch
from PINN_Net import PINN_Model, PINN_Dataset
import pandas as pd
from visdom import Visdom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev_size = 37 * 40
dev_dataset = PINN_Dataset(file_path='', device=device, require_grad=False)
devloader = DataLoader(dev_dataset, batch_size=dev_size, shuffle=False, drop_last=False)
model = PINN_Model(device)
model.load_state_dict(torch.load(''))
model.to(device)

dev_node = pd.read_excel('', sheet_name='')
dev_node = dev_node.drop('Time', axis=1)
dev_node = torch.tensor(np.array(dev_node)).to(device)
dev_node = dev_node[1:, [1, 4, 7, 10, 13, 16]]

dev_capacity = pd.read_excel('', sheet_name='')
dev_capacity = dev_capacity.drop('Time', axis=1)
dev_capacity = torch.tensor(np.array(dev_capacity)).to(device)
dev_capacity = dev_capacity[1:, [4, 13]]
dev_velocity = pd.read_excel('', sheet_name='')
dev_velocity = dev_velocity.drop('Time', axis=1)
dev_velocity = torch.tensor(np.array(dev_velocity)).to(device)
dev_velocity = dev_velocity[1:, [4, 13]]

def PINN_Test(model, devloader, device, dev_node, dev_capacity, dev_velocity):
    import time

    # viz = Visdom(env='0425_PINN_Test')
    node_num = 19
    edge_num = 18
    model.eval()
    all_node_out = torch.zeros((863, node_num)).to(device)
    all_edge_out = torch.zeros((863, edge_num, 2)).to(device)
    all_size = 0
    node_name = ['2', '5', '8', '11', '14', '17']
    edge_name = ['5_6', '14_15']

    X1 = 0.142857142857143
    X2 = 0.257142857142857
    X3 = 0.582142857142857
    X4 = 0.692857142857143
    L1 = 0.171428571428571
    L2 = 0.276785714285714
    time_all = []
    for index, dev_data in enumerate(devloader, 0):
        start = time.time()
        x, t, q, node_id, D, S0, n, before_h, before_u, target = dev_data
        before_h[torch.where(x == 0.2)] = ((before_h[torch.where(x == X2)] - before_h[torch.where(x == X1)]) / (X2 - X1)
                                           + before_h[torch.where(x == X1)])

        before_h[torch.where(x == 0.628571428571429)] = ((before_h[torch.where(x == X4)] - before_h[torch.where(x == X3)])
                                                         / (X4 - X3) + before_h[torch.where(x == X3)])
        before_h[torch.where(x == 0.228571428571429)] = ((before_h[torch.where(x == L2)] - before_h[torch.where(x == L1)]) / (L2 - L1) + before_h[torch.where(x == L1)])
        before_u[torch.where(x == 0.2)] = ((before_u[torch.where(x == X2)] - before_u[torch.where(x == X1)]) / (X2 - X1) + before_u[torch.where(x == X1)])
        before_u[torch.where(x == 0.521428571428571)] = ((before_u[torch.where(x == X4)] - before_u[torch.where(x == X3)]) / (X4 - X3) + before_u[torch.where(x == X3)])
        before_u[torch.where(x == 0.228571428571429)] = ((before_u[torch.where(x == L2)] - before_u[torch.where(x == L1)]) / (L2 - L1) + before_u[torch.where(x == L1)])

        h_w, u, lambda1, lambda2 = model(x, t, q, node_id, before_h, before_u)

        out = torch.cat((x, t, h_w, u, node_id), dim=1)
        node_out, edge_out = data_retransfer(out, device)
        size = node_out.shape[0]
        all_node_out[all_size:all_size + size, :] = node_out
        all_edge_out[all_size:all_size + size, :, :] = edge_out
        all_size += size
        end = time.time()
        time_all.append(end - start)
    test_node_out = all_node_out[:, [1, 4, 7, 10, 13, 16]]
    test_edge_out = all_edge_out[:, [4, 13], :]

    print('Time:', sum(time_all))
    nse_node = nse_loss(test_node_out, dev_node).item()
    nse_edge_capacity = nse_loss(test_edge_out[:, :, 0], dev_capacity).item()
    nse_edge_velocity = nse_loss(test_edge_out[:, :, 1], dev_velocity).item()
    nse = (nse_node + nse_edge_capacity + nse_edge_velocity) / 3
    min_nse = min(nse_node, nse_edge_capacity, nse_edge_velocity)
    for i in range(dev_node.shape[1]):
        viz.line(test_node_out[:, i].cpu().detach().numpy(), [time for time in range(863)], win='NODE_' + str(i),
                 name='Pred', update='append', opts=dict(title='NODE_' + node_name[i]))
        viz.line(dev_node[:, i].cpu().detach().numpy(), [time for time in range(863)], win='NODE_' + str(i),
                 name='True', update='append', opts=dict(title='NODE_' + node_name[i]))
        print('NODE_', node_name[i], 'NSE: ', nse_loss(test_node_out[:, i], dev_node[:, i]).item())
    print(test_edge_out.shape)
    for i in range(dev_capacity.shape[1]):
        viz.line(test_edge_out[:, i, 0].cpu().detach().numpy(), [time for time in range(863)], win='EDGE_Capacity_' + str(i),
                 name='Pred', update='append', opts=dict(title='EDGE_Capacity_' + edge_name[i]))
        viz.line(dev_capacity[:, i].cpu().detach().numpy(), [time for time in range(863)], win='EDGE_Capacity_' + str(i),
                 name='True', update='append', opts=dict(title='EDGE_Capacity_' + edge_name[i]))
        print('EDGE_', edge_name[i], 'Capacity NSE: ', nse_loss(test_edge_out[:, i, 0], dev_capacity[:, i]).item())
    for i in range(dev_velocity.shape[1]):
        viz.line(test_edge_out[:, i, 1].cpu().detach().numpy(), [time for time in range(863)], win='EDGE_Velocity_' + str(i),
                 name='Pred', update='append', opts=dict(title='EDGE_Velocity_' + edge_name[i]))
        viz.line(dev_velocity[:, i].cpu().detach().numpy(), [time for time in range(863)], win='EDGE_Velocity_' + str(i),
                 name='True', update='append', opts=dict(title='EDGE_Velocity_' + edge_name[i]))
        print('EDGE_', edge_name[i], 'Velocity NSE: ', nse_loss(test_edge_out[:, i, 1], dev_velocity[:, i]).item())
    all_node_out = pd.DataFrame(all_node_out.cpu().detach().numpy())
    all_node_out.to_csv('Predict_Node_PINN.csv', index=False)
    capacity_out= pd.DataFrame(all_edge_out[:, :, 0].cpu().detach().numpy())
    capacity_out.to_csv('Predict_Capacity_PINN.csv', index=False)
    velocity_out = pd.DataFrame(all_edge_out[:, :, 1].cpu().detach().numpy())
    velocity_out.to_csv('Predict_Velocity_PINN.csv', index=False)
    return nse, min_nse
nse, min_nse = PINN_Test(model, devloader, device, dev_node, dev_capacity, dev_velocity)
print(nse, min_nse)
import numpy as np
from Data_ReTransfer import data_retransfer
from Function import setup_seed, weight_init, nse_loss
from torch.utils.data import DataLoader
import torch
from PINN_Net import PINN_Model, PINN_Dataset
import pandas as pd
from visdom import Visdom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 37 * 64
dev_size = 37 * 863
lr = 1e-3
weight_decay = 1e-5
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

setup_seed(14)
model = PINN_Model(device=device)
model.apply(weight_init)

train_dataset = PINN_Dataset(file_path='', device=device, require_grad=True)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
dev_dataset = PINN_Dataset(file_path='', device=device, require_grad=False)
devloader = DataLoader(dev_dataset, batch_size=dev_size, shuffle=False, drop_last=False)

def PINN_Train(model, dataloader, devloader, max_epoch, device, lr, weight_decay):
    import time

    start_time = time.time()
    time_all = []
    end_all = []
    pde_all = []
    data_all = []
    loss_all = []
    Criterion_Data = torch.nn.MSELoss(reduction='mean').to(device)
    Criterion_PDE = torch.nn.L1Loss(reduction='mean').to(device)
    Optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay,
                                  amsgrad=False)

    model = model.to(device)
    torch.save(model.state_dict(), 'PINN_'+str(0)+'.pth')
    viz = Visdom(env='')
    draw_window1 = 'TotalLoss'
    draw_window2 = 'Data_Loss'
    draw_window3 = 'PDE_Loss'
    draw_time = 0
    node_num = 19
    edge_num = 18
    lowerest_nse = -np.inf
    mean_nse = -np.inf
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(max_epoch):
        model.eval()
        all_node_out = torch.zeros((863, node_num)).to(device)
        all_edge_out = torch.zeros((863, edge_num, 2)).to(device)
        all_size = 0
        for index, dev_data in enumerate(devloader, 0):
            x, t, q, node_id, D, S0, n, before_h, before_u, target = dev_data
            h_w, u, lambda1, lambda2 = model(x, t, q, node_id, before_h, before_u)
            lambda_1_copy = lambda1.cpu().detach().numpy()
            lambda_2_copy = lambda2.cpu().detach().numpy()
            out = torch.cat((x, t, h_w, u, node_id), dim=1)
            node_out, edge_out = data_retransfer(out, device)
            size = node_out.shape[0]
            all_node_out[all_size:all_size + size, :] = node_out
            all_edge_out[all_size:all_size + size, :, :] = edge_out
            all_size += size

        test_node_out = all_node_out[:, [1, 4, 7, 10, 13, 16]]
        test_edge_out = all_edge_out[:, [4, 13], :]
        nse_node = nse_loss(test_node_out, dev_node).item()
        nse_edge_capacity = nse_loss(test_edge_out[:, :, 0], dev_capacity).item()
        nse_edge_velocity = nse_loss(test_edge_out[:, :, 1], dev_velocity).item()
        nse = (nse_node + nse_edge_capacity + nse_edge_velocity) / 3
        min_nse = min(nse_node, nse_edge_capacity, nse_edge_velocity)
        if nse > mean_nse:
            mean_nse = nse
        if min_nse > lowerest_nse:
            lowerest_nse = min_nse

        viz.line([lambda_1_copy * lambda_2_copy], [epoch], win='lambda', name='data_node', update='append')
        viz.line([lambda_1_copy * (1 - lambda_2_copy)], [epoch], win='lambda', name='data_edge', update='append')
        viz.line([(1 - lambda_1_copy)], [epoch], win='lambda', name='pde', update='append')

        if epoch == 0:
            viz.line([nse_node], [0],
                     win='NODE_NSE', name='NODE_NSE', update='append')
            viz.line([nse_edge_capacity], [0],
                     win='EDGE_w_NSE', name='EDGE_w_NSE', update='append')
            viz.line([nse_edge_velocity], [0],
                     win='EDGE_u_NSE', name='EDGE_u_NSE', update='append')
            viz.line(test_node_out[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)], win='NODE_5',
                     name='Pred', update='append', opts=dict(title='NODE_5'))
            viz.line(dev_node[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)], win='NODE_5',
                     name='True', update='append', opts=dict(title='NODE_5'))
            viz.line(test_edge_out[:, 0, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Capacity',
                     name='Pred', update='append', opts=dict(title='EDGE_5_6_Capacity'))
            viz.line(dev_capacity[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Capacity',
                     name='True', update='append', opts=dict(title='EDGE_5_6_Capacity'))
            viz.line(test_edge_out[:, 0, 1].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Velocity',
                     name='Pred', update='append', opts=dict(title='EDGE_5_6_Velocity'))
            viz.line(dev_velocity[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Velocity',
                     name='True', update='append', opts=dict(title='EDGE_5_6_Velocity'))
        else:
            viz.line([nse_node], [epoch],
                     win='NODE_NSE', name='NODE_NSE', update='append')
            viz.line([nse_edge_capacity], [epoch],
                     win='EDGE_w_NSE', name='EDGE_w_NSE', update='append')
            viz.line([nse_edge_velocity], [epoch],
                     win='EDGE_u_NSE', name='EDGE_u_NSE', update='append')
            viz.line(test_node_out[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)], win='NODE_5',
                     name='Pred', update='replace', opts=dict(title='NODE_5'))
            viz.line(dev_node[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)], win='NODE_5',
                     name='True', update='replace', opts=dict(title='NODE_5'))
            viz.line(test_edge_out[:, 0, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Capacity',
                     name='Pred', update='replace', opts=dict(title='EDGE_5_6_Capacity'))
            viz.line(dev_capacity[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Capacity',
                     name='True', update='replace', opts=dict(title='EDGE_5_6_Capacity'))
            viz.line(test_edge_out[:, 0, 1].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Velocity',
                     name='Pred', update='replace', opts=dict(title='EDGE_5_6_Velocity'))
            viz.line(dev_velocity[:, 0].cpu().detach().numpy(), [draw_time for draw_time in range(863)],
                     win='EDGE_5_6_Velocity',
                     name='True', update='replace', opts=dict(title='EDGE_5_6_Velocity'))

        model.train()
        train_loss = 0
        data_train_loss = 0
        pde_train_loss = 0
        for index, data in enumerate(dataloader, 0):
            x, t, q, node_id, D, S0, n, before_h, before_u, target = data
            h_w, u, lambda1, lambda2 = model(x, t, q, node_id, before_h, before_u)
            test_gradient_func = torch.zeros((x.shape[0], 1)).to(device)
            dhdx = torch.autograd.grad(outputs=h_w, inputs=x, retain_graph=True, grad_outputs=torch.ones_like(h_w), create_graph=True)[0]
            dudx = torch.autograd.grad(outputs=u, inputs=x, retain_graph=True, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            dhdt = torch.autograd.grad(outputs=h_w, inputs=t, retain_graph=True, grad_outputs=torch.ones_like(h_w), create_graph=True)[0]
            f1_node = D * dhdt + D * u * dhdx + D * h_w * dudx - q
            f1 = f1_node * node_id

            data_loss_1 = Criterion_Data(torch.concatenate((h_w, u), dim=1) * node_id, target * node_id)
            data_loss_2 = Criterion_Data(torch.concatenate((h_w, u), dim=1) * (1 - node_id), target * (1 - node_id))
            data_loss = data_loss_1 * lambda2 + data_loss_2 * (1 - lambda2)
            pde_loss = Criterion_PDE(f1, test_gradient_func)
            Loss = lambda1 * (data_loss_1 * lambda2 + data_loss_2 * (1 - lambda2)) + (1 - lambda1) * pde_loss
            # Backprop the loss
            Loss.backward()
            # Update model parameters
            Optimizer.step()
            # Clear gradients for next step
            Optimizer.zero_grad()

            loss1_copy = Loss.cpu().detach().numpy()
            loss2_copy = data_loss.cpu().detach().numpy()
            loss3_copy = pde_loss.cpu().detach().numpy()
            if draw_time == 0:
                viz.line([loss1_copy], [0], win=draw_window1, opts=dict(title=draw_window1))
                viz.line([loss2_copy], [0], win=draw_window2, opts=dict(title=draw_window2))
                viz.line([loss3_copy], [0], win=draw_window3, opts=dict(title=draw_window3))
                draw_time += 1
            elif index % 10 == 0:
                viz.line([loss1_copy], [draw_time], win=draw_window1, update='append')
                viz.line([loss2_copy], [draw_time], win=draw_window2, update='append')
                viz.line([loss3_copy], [draw_time], win=draw_window3, update='append')
                draw_time += 1
            train_loss += loss1_copy
            data_train_loss += loss2_copy
            pde_train_loss += loss3_copy
            loss_all.append(loss1_copy)
            data_all.append(loss2_copy)
            pde_all.append(loss3_copy)
            end_time = time.time()
            time_all.append(end_time - start_time)
        print('Epoch: %d, Loss: %.5f, Data_Loss: %.5f, PDE_Loss: %.5f' % (epoch, train_loss, data_train_loss, pde_train_loss))
        torch.save(model.state_dict(), 'PINN_' + str(epoch + 1) + '.pth')
    time_all = pd.DataFrame(time_all)
    time_all.to_csv('time_all.csv', index=False)
    loss_all = pd.DataFrame(loss_all)
    loss_all.to_csv('loss_all.csv', index=False)
    data_all = pd.DataFrame(data_all)
    data_all.to_csv('data_all.csv', index=False)
    pde_all = pd.DataFrame(pde_all)
    pde_all.to_csv('pde_all.csv', index=False)
    return model

max_epoch = 300
model = PINN_Train(model, dataloader, devloader, max_epoch, device, lr, weight_decay)
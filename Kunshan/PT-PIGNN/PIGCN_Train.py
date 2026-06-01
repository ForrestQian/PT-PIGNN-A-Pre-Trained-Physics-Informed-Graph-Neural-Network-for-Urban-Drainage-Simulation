import pandas as pd
import numpy as np
import torch
from Function import setup_seed, weight_init, PDE_Loss, nse_loss, subsets, subsets_increasing, subsets_decreasing, Area, inverse_f, Area_np
from PIGCN_Net import GCN_Dataset, GCN_Model
from torch.utils.data import DataLoader
import gc
from visdom import Visdom
import time
def PIGCN_train(length_index, dev_size, train_node_index, train_edge_index,                          ## training nodes
                dev_node_index, dev_edge_index,
                D_input, n_input, S0_input,                                                    ## physical properties
                epochs, physical_pre, physical_epochs,
                model, batch_size, device, lr, weight_decay,                                 ## training hyperparameters
                edge_index, dataloader, devloader):
    ## data
    viz = Visdom(env='Train_PrePiGCN_Data')
    time_all = []
    Loss_all = []
    Loss_1_all = []
    Loss_2_all = []
    PDE_1_all = []
    PDE_2_all = []
    start_time = time.time()
    nse_all = pd.DataFrame(columns=['Node 2', 'Node 5', 'Node 8', 'Node 11', 'Node 14', 'Node 17',
                                'Link 5-6 Flow', 'Link 5-6 Velocity', 'Link 14-15 Flow', 'Link 14-15 Velocity'])
    mae_all = pd.DataFrame(columns=['Node 2', 'Node 5', 'Node 8', 'Node 11', 'Node 14', 'Node 17',
                                'Link 5-6 Flow', 'Link 5-6 Velocity', 'Link 14-15 Flow', 'Link 14-15 Velocity'])
    rmse_all = pd.DataFrame(columns=['Node 2', 'Node 5', 'Node 8', 'Node 11', 'Node 14', 'Node 17',
                                'Link 5-6 Flow', 'Link 5-6 Velocity', 'Link 14-15 Flow', 'Link 14-15 Velocity'])
    mape_all = pd.DataFrame(columns=['Node 2', 'Node 5', 'Node 8', 'Node 11', 'Node 14', 'Node 17',
                                'Link 5-6 Flow', 'Link 5-6 Velocity', 'Link 14-15 Flow', 'Link 14-15 Velocity'])
    file_path = ''
    node_data_h = pd.read_excel(file_path, sheet_name='Node_Depth')
    node_data_h = np.array(node_data_h)[:, 1:].astype(np.float32)
    max_h = np.max(node_data_h, axis=0)
    min_h = np.min(node_data_h, axis=0)
    max_h = torch.tensor(max_h, requires_grad=True).to(device)
    max_h = torch.max(max_h).detach()
    min_h = torch.tensor(min_h, requires_grad=True).to(device)
    min_h = torch.min(min_h).detach()

    node_data_q = pd.read_excel(file_path, sheet_name='Node_Inflow')
    node_data_q = np.array(node_data_q)[:, 1:].astype(np.float32)
    max_q = np.max(node_data_q, axis=0)
    min_q = np.min(node_data_q, axis=0)
    max_q = torch.tensor(max_q, requires_grad=True).to(device)
    max_q = torch.max(max_q).detach()
    min_q = torch.tensor(min_q, requires_grad=True).to(device)
    min_q = torch.min(min_q).detach()

    edge_data_velocity = pd.read_excel(file_path, sheet_name='Link_Velocity')
    edge_data_velocity = np.array(edge_data_velocity)[:, 1:].astype(np.float32)
    max_velocity = np.max(edge_data_velocity, axis=0)
    min_velocity = np.min(edge_data_velocity, axis=0)

    full_max_velocity = []
    full_min_velocity = []
    for i in range(len(max_velocity)):
        full_max_velocity.extend([max_velocity[i]] * length_index[i])
        full_min_velocity.extend([min_velocity[i]] * length_index[i])
    full_max_velocity = torch.tensor(full_max_velocity, requires_grad=True).to(device).detach()
    full_min_velocity = torch.tensor(full_min_velocity, requires_grad=True).to(device).detach()

    max_velocity = torch.tensor(max_velocity, requires_grad=True).to(device)
    max_velocity = torch.max(max_velocity).detach()
    min_velocity = torch.tensor(min_velocity, requires_grad=True).to(device)
    min_velocity = torch.min(min_velocity).detach()

    edge_data_flow = pd.read_excel(file_path, sheet_name='Link_Flow')
    edge_data_flow = Area_np(np.array(edge_data_flow)[:, 1:].astype(np.float32), D=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])) * edge_data_velocity
    max_flow = np.max(edge_data_flow, axis=0).astype(np.float32)
    min_flow = np.min(edge_data_flow, axis=0).astype(np.float32)

    full_max_flow = []
    full_min_flow = []
    for i in range(len(max_flow)):
        full_max_flow.extend([max_flow[i]] * length_index[i])
        full_min_flow.extend([min_flow[i]] * length_index[i])
    full_max_flow = torch.tensor(full_max_flow, requires_grad=True).to(device).detach()
    full_min_flow = torch.tensor(full_min_flow, requires_grad=True).to(device).detach()

    max_flow = torch.tensor(max_flow, requires_grad=True).to(device)
    max_flow = torch.max(max_flow).detach()
    min_flow = torch.tensor(min_flow, requires_grad=True).to(device)
    min_flow = torch.min(min_flow).detach()

    setup_seed(0)
    test_size = 864 - 1
    num_edges = len(length_index)
    D_feature = torch.zeros((batch_size, sum(length_index), 301))
    D_origin_train = torch.zeros((batch_size, len(length_index)))
    D_origin_dev = torch.zeros((test_size, len(length_index)))
    D_feature_dev = torch.zeros((dev_size, sum(length_index)))
    S0_feature = torch.zeros((batch_size, sum(length_index), 301))
    S0_feature_dev = torch.zeros((dev_size, sum(length_index)))
    n_feature = torch.zeros((batch_size, sum(length_index), 301))
    n_feature_dev = torch.zeros((dev_size, sum(length_index)))



    for nn in range(num_edges):
        start_sum = sum(length_index[:nn])
        now_sum = sum(length_index[:nn + 1])
        D_feature[:, start_sum:now_sum, :] = D_input[nn]
        D_feature_dev[:, start_sum:now_sum] = D_input[nn]
        D_origin_dev[:, nn] = D_input[nn]
        D_origin_train[:, nn] = D_input[nn]
        S0_feature[:, start_sum:now_sum, :] = S0_input[nn]
        S0_feature_dev[:, start_sum:now_sum] = S0_input[nn]
        n_feature[:, start_sum:now_sum, :] = n_input[nn]
        n_feature_dev[:, start_sum:now_sum] = n_input[nn]
        start_sum += now_sum
    D_feature.requires_grad = True
    S0_feature.requires_grad = True
    n_feature.requires_grad = True
    D_feature = D_feature.to(device)
    D_origin_train = D_origin_train.to(device).detach()
    S0_feature = S0_feature.to(device).detach()
    n_feature = n_feature.to(device).detach()
    D_input = torch.tensor(D_input).requires_grad_(True).to(device)
    g = 9.8

    # Reset loss bias weights
    model.lambda1 = torch.nn.Parameter(torch.tensor([0.50]))
    model.lambda2 = torch.nn.Parameter(torch.tensor([0.50]))
    model.lambda3 = torch.nn.Parameter(torch.tensor([0.50]))
    model = weight_init(model)
    # Load model
    model.load_state_dict(torch.load(''))
    model = model.to(device)

    PDE_Label_1 = torch.zeros((batch_size, sum(length_index), 301), requires_grad=True).to(device)
    PDE_Label_2 = torch.zeros((batch_size, sum(length_index), 301), requires_grad=True).to(device)
    Criterion_PDE = torch.nn.L1Loss(reduction='mean').to(device)
    Criterion_DATA = torch.nn.MSELoss(reduction='mean').to(device)
    Optimizer_PDE = torch.optim.AdamW(model.parameters(), lr=lr[0], betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay[0], amsgrad=False)
    Optimizer_DATA = torch.optim.AdamW(model.parameters(), lr=lr[1], betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay[1], amsgrad=False)

    torch.autograd.set_detect_anomaly(True)

    # Decrement indices by one
    train_node = [i - 1 for i in train_node_index]
    dev_node = [i - 1 for i in dev_node_index]
    train_edge = [i - 1 for i in train_edge_index]
    dev_edge = [i - 1 for i in dev_edge_index]

    draw_time = 0
    for i in range(epochs):
        print('Start_Epoch:', i)

        all_output_edge = torch.zeros(test_size, num_edges, 2)
        all_output_node = torch.zeros(test_size, num_nodes)

        all_test_edge = torch.zeros(test_size, num_edges, 2)
        all_test_node = torch.zeros(test_size, num_nodes)

        for index, data in enumerate(devloader, 0):
            model.eval()
            node_feature, edge_feature, output_node_feature, output_edge_feature = data
            node_feature = node_feature.to(device)
            edge_feature = edge_feature.to(device)
            output_node_feature = output_node_feature.to(device)
            output_edge_feature = output_edge_feature.to(device)

            node_updated, edge_updated, expanded_e, lambda1, lambda2, lambda3 = model(node_feature, edge_feature, D_input)
            all_test_edge[index * dev_size: index * dev_size + node_feature.shape[0], :, :] = edge_updated.cpu().detach()
            all_test_node[index * dev_size: index * dev_size + node_feature.shape[0], :] = node_updated.cpu().detach()
            all_output_node[index * dev_size: index * dev_size + node_feature.shape[0], :] = output_node_feature.cpu().detach()
            all_output_edge[index * dev_size: index * dev_size + node_feature.shape[0], :, :] = output_edge_feature.cpu().detach()

        all_output_edge[:, :, 0] = Area(all_output_edge[:, :, 0], D_origin_dev) * all_output_edge[:, :, 1]
        all_test_edge[:, :, 0] = Area(all_test_edge[:, :, 0], D_origin_dev) * all_test_edge[:, :, 1]
        # if physical_pre:
        #     all_test_node = all_test_node
        all_output_node = all_output_node[:, dev_node]
        all_output_edge = all_output_edge[:, dev_edge, :]
        new_node_updated = all_test_node[:, dev_node]
        new_edge_updated = all_test_edge[:, dev_edge, :]

        nse_node = nse_loss(all_output_node, new_node_updated)
        mae_node = torch.mean(torch.abs(all_output_node - new_node_updated), dim=0)
        rmse_node = torch.sqrt(torch.mean((all_output_node - new_node_updated) ** 2, dim=0))
        mape_node = torch.mean(torch.abs(all_output_node - new_node_updated) / all_output_node, dim=0)

        nse_edge_capacity = nse_loss(all_output_edge[:, :, 0], new_edge_updated[:, :, 0])
        mae_edge_capacity = torch.mean(torch.abs(all_output_edge[:, :, 0] - new_edge_updated[:, :, 0]), dim=0)
        rmse_edge_capacity = torch.sqrt(torch.mean((all_output_edge[:, :, 0] - new_edge_updated[:, :, 0]) ** 2, dim=0))
        mape_edge_capacity = torch.mean(torch.abs(all_output_edge[:, :, 0] - new_edge_updated[:, :, 0]) / all_output_edge[:, :, 0], dim=0)

        nse_edge_velocity = nse_loss(all_output_edge[:, :, 1], new_edge_updated[:, :, 1])
        mae_edge_velocity = torch.mean(torch.abs(all_output_edge[:, :, 1] - new_edge_updated[:, :, 1]), dim=0)
        rmse_edge_velocity = torch.sqrt(torch.mean((all_output_edge[:, :, 1] - new_edge_updated[:, :, 1]) ** 2, dim=0))
        mape_edge_velocity = torch.mean(torch.abs(all_output_edge[:, :, 1] - new_edge_updated[:, :, 1]) / all_output_edge[:, :, 1], dim=0)
        nse_all.loc[i] = [nse_node[0].item(), nse_node[1].item(), nse_node[2].item(), nse_node[3].item(), nse_node[4].item(), nse_node[5].item(),
                            nse_edge_capacity[0].item(), nse_edge_velocity[0].item(), nse_edge_capacity[1].item(), nse_edge_velocity[1].item()]
        mae_all.loc[i] = [mae_node[0].item(), mae_node[1].item(), mae_node[2].item(), mae_node[3].item(), mae_node[4].item(), mae_node[5].item(),
                            mae_edge_capacity[0].item(), mae_edge_velocity[0].item(), mae_edge_capacity[1].item(), mae_edge_velocity[1].item()]
        rmse_all.loc[i] = [rmse_node[0].item(), rmse_node[1].item(), rmse_node[2].item(), rmse_node[3].item(), rmse_node[4].item(), rmse_node[5].item(),
                            rmse_edge_capacity[0].item(), rmse_edge_velocity[0].item(), rmse_edge_capacity[1].item(), rmse_edge_velocity[1].item()]
        mape_all.loc[i] = [mape_node[0].item(), mape_node[1].item(), mape_node[2].item(), mape_node[3].item(), mape_node[4].item(), mape_node[5].item(),
                            mape_edge_capacity[0].item(), mape_edge_velocity[0].item(), mape_edge_capacity[1].item(), mape_edge_velocity[1].item()]


        print(nse_node)
        print(nse_edge_capacity)
        print(nse_edge_velocity)

        gc.collect()
        torch.cuda.empty_cache()
        if draw_time == 0:
            viz.line([nse_node[0].cpu().detach().numpy()], [0], win='NODE_2_NSE', name='Pred',
                     opts=dict(title='NODE_2_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[1].cpu().detach().numpy()], [0], win='NODE_5_NSE', name='Pred',
                     opts=dict(title='NODE_5_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[2].cpu().detach().numpy()], [0], win='NODE_8_NSE', name='Pred',
                     opts=dict(title='NODE_8_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[3].cpu().detach().numpy()], [0], win='NODE_11_NSE', name='Pred',
                     opts=dict(title='NODE_11_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[4].cpu().detach().numpy()], [0], win='NODE_14_NSE', name='Pred',
                     opts=dict(title='NODE_14_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[5].cpu().detach().numpy()], [0], win='NODE_17_NSE', name='Pred',
                     opts=dict(title='NODE_17_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))

            viz.line([nse_edge_capacity[0].cpu().detach().numpy()], [0], win='Link_5_6_Flow_NSE',
                     name='Pred', opts=dict(title='Link_5_6_Flow_NSE', linecolor=np.array([[0, 0, 255]]),
                     dash=np.array(['solid'])))
            viz.line([nse_edge_velocity[0].cpu().detach().numpy()], [draw_time], win='Link_5_6_Velocity_NSE',
                     name='Pred', opts=dict(title='Link_5_6_Velocity_NSE', linecolor=np.array([[0, 0, 255]]),
                     dash=np.array(['solid'])))
            viz.line([nse_edge_capacity[1].cpu().detach().numpy()], [draw_time], win='Link_14_15_Flow_NSE',
                     name='Pred', opts=dict(title='Link_14_15_Flow_NSE', linecolor=np.array([[0, 0, 255]]),
                     dash=np.array(['solid'])))
            viz.line([nse_edge_velocity[1].cpu().detach().numpy()], [draw_time], win='Link_14_15_Velocity_NSE',
                     name='Pred', opts=dict(title='Link_14_15_Velocity_NSE', linecolor=np.array([[0, 0, 255]]),
                     dash=np.array(['solid'])))

            viz.line(new_node_updated[:, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_2',
                     name='Pred', update='append', opts=dict(title='NODE_2'))
            viz.line(all_output_node[:, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_2',
                     name='True', update='append', opts=dict(title='NODE_2'))
            viz.line(new_node_updated[:, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_5',
                     name='Pred', update='append', opts=dict(title='NODE_5'))
            viz.line(all_output_node[:, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_5',
                     name='True', update='append', opts=dict(title='NODE_5'))
            viz.line(new_node_updated[:, 2].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_8',
                     name='Pred', update='append', opts=dict(title='NODE_8'))
            viz.line(all_output_node[:, 2].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_8',
                     name='True', update='append', opts=dict(title='NODE_8'))
            viz.line(new_node_updated[:, 3].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_11',
                     name='Pred', update='append', opts=dict(title='NODE_11'))
            viz.line(all_output_node[:, 3].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_11',
                     name='True', update='append', opts=dict(title='NODE_11'))
            viz.line(new_node_updated[:, 4].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_14',
                     name='Pred', update='append', opts=dict(title='NODE_14'))
            viz.line(all_output_node[:, 4].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_14',
                     name='True', update='append', opts=dict(title='NODE_14'))
            viz.line(new_node_updated[:, 5].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_17',
                     name='Pred', update='append', opts=dict(title='NODE_17'))
            viz.line(all_output_node[:, 5].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='NODE_17',
                     name='True', update='append', opts=dict(title='NODE_17'))

            viz.line(new_edge_updated[:, 0, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_5_6_Flow',
                     name='Pred', update='append', opts=dict(title='Link_5_6_Flow'))
            viz.line(all_output_edge[:, 0, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_5_6_Flow',
                     name='True', update='append', opts=dict(title='Link_5_6_Flow'))
            viz.line(new_edge_updated[:, 0, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_5_6_Velocity',
                     name='Pred', update='append', opts=dict(title='Link_5_6_Velocity'))
            viz.line(all_output_edge[:, 0, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_5_6_Velocity',
                     name='True', update='append', opts=dict(title='Link_5_6_Velocity'))
            viz.line(new_edge_updated[:, 1, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_14_15_Flow',
                     name='Pred', update='append', opts=dict(title='Link_14_15_Flow'))
            viz.line(all_output_edge[:, 1, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_14_15_Flow',
                     name='True', update='append', opts=dict(title='Link_14_15_Flow'))
            viz.line(new_edge_updated[:, 1, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_14_15_Velocity',
                     name='Pred', update='append', opts=dict(title='Link_14_15_Velocity'))
            viz.line(all_output_edge[:, 1, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                     win='Link_14_15_Velocity',
                     name='True', update='append', opts=dict(title='Link_14_15_Velocity'))
            draw_time = draw_time + 1
        else:
            viz.line([nse_node[0].cpu().detach().numpy()], [draw_time], win='NODE_2_NSE', name='Pred', update='append', opts=dict(title='NODE_2_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[1].cpu().detach().numpy()], [draw_time], win='NODE_5_NSE', name='Pred', update='append', opts=dict(title='NODE_5_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[2].cpu().detach().numpy()], [draw_time], win='NODE_8_NSE', name='Pred', update='append', opts=dict(title='NODE_8_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[3].cpu().detach().numpy()], [draw_time], win='NODE_11_NSE', name='Pred', update='append', opts=dict(title='NODE_11_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[4].cpu().detach().numpy()], [draw_time], win='NODE_14_NSE', name='Pred', update='append', opts=dict(title='NODE_14_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_node[5].cpu().detach().numpy()], [draw_time], win='NODE_17_NSE', name='Pred', update='append', opts=dict(title='NODE_17_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))

            viz.line([nse_edge_capacity[0].cpu().detach().numpy()], [draw_time], win='Link_5_6_Flow_NSE', name='Pred', update='append', opts=dict(title='Link_5_6_Flow_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_edge_velocity[0].cpu().detach().numpy()], [draw_time], win='Link_5_6_Velocity_NSE', name='Pred', update='append', opts=dict(title='Link_5_6_Velocity_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_edge_capacity[1].cpu().detach().numpy()], [draw_time], win='Link_14_15_Flow_NSE', name='Pred', update='append', opts=dict(title='Link_14_15_Flow_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
            viz.line([nse_edge_velocity[1].cpu().detach().numpy()], [draw_time], win='Link_14_15_Velocity_NSE', name='Pred', update='append', opts=dict(title='Link_14_15_Velocity_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))

            viz.line(new_node_updated[:, 0].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_2',
                     name='Pred', update='replace', opts=dict(title='NODE_2'))
            viz.line(all_output_node[:, 0].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_2',
                     name='True', update='replace', opts=dict(title='NODE_2'))
            viz.line(new_node_updated[:, 1].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_5',
                     name='Pred', update='replace', opts=dict(title='NODE_5'))
            viz.line(all_output_node[:, 1].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_5',
                     name='True', update='replace', opts=dict(title='NODE_5'))
            viz.line(new_node_updated[:, 2].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_8',
                     name='Pred', update='replace', opts=dict(title='NODE_8'))
            viz.line(all_output_node[:, 2].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_8',
                     name='True', update='replace', opts=dict(title='NODE_8'))
            viz.line(new_node_updated[:, 3].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_11',
                     name='Pred', update='replace', opts=dict(title='NODE_11'))
            viz.line(all_output_node[:, 3].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_11',
                     name='True', update='replace', opts=dict(title='NODE_11'))
            viz.line(new_node_updated[:, 4].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_14',
                     name='Pred', update='replace', opts=dict(title='NODE_14'))
            viz.line(all_output_node[:, 4].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_14',
                     name='True', update='replace', opts=dict(title='NODE_14'))
            viz.line(new_node_updated[:, 5].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_17',
                     name='Pred', update='replace', opts=dict(title='NODE_17'))
            viz.line(all_output_node[:, 5].cpu().detach().numpy(), [time for time in range(test_size)], win='NODE_17',
                     name='True', update='replace', opts=dict(title='NODE_17'))

            viz.line(new_edge_updated[:, 0, 0].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_5_6_Flow',
                     name='Pred', update='replace', opts=dict(title='Link_5_6_Flow'))
            viz.line(all_output_edge[:, 0, 0].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_5_6_Flow',
                     name='True', update='replace', opts=dict(title='Link_5_6_Flow'))
            viz.line(new_edge_updated[:, 0, 1].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_5_6_Velocity',
                     name='Pred', update='replace', opts=dict(title='Link_5_6_Velocity'))
            viz.line(all_output_edge[:, 0, 1].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_5_6_Velocity',
                     name='True', update='replace', opts=dict(title='Link_5_6_Velocity'))
            viz.line(new_edge_updated[:, 1, 0].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_14_15_Flow',
                     name='Pred', update='replace', opts=dict(title='Link_14_15_Flow'))
            viz.line(all_output_edge[:, 1, 0].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_14_15_Flow',
                     name='True', update='replace', opts=dict(title='Link_14_15_Flow'))
            viz.line(new_edge_updated[:, 1, 1].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_14_15_Velocity',
                     name='Pred', update='replace', opts=dict(title='Link_14_15_Velocity'))
            viz.line(all_output_edge[:, 1, 1].cpu().detach().numpy(), [time for time in range(test_size)], win='Link_14_15_Velocity',
                     name='True', update='replace', opts=dict(title='Link_14_15_Velocity'))
            draw_time = draw_time + 1

        if physical_pre:

            for index_physical in range(physical_epochs):
                model.train()
                # Generate boundary conditions arbitrarily
                h_feature = torch.rand(batch_size, num_nodes).to(device)
                h_feature = h_feature.unsqueeze(dim=2)
                q_feature = torch.rand(batch_size, num_nodes).to(device)
                q_feature = q_feature.unsqueeze(dim=2)
                node_feature = torch.concatenate((h_feature, q_feature), dim=2)

                flow_feature = torch.rand(batch_size, num_edges).to(device)
                flow_feature = flow_feature.unsqueeze(dim=2)
                velocity_feature = torch.rand(batch_size, num_edges).to(device)
                velocity_feature = velocity_feature.unsqueeze(dim=2)
                edge_feature = torch.concatenate((flow_feature, velocity_feature), dim=2)
                node_feature[:, :, 0] = node_feature[:, :, 0] * (max_h - min_h + 0.0001) + min_h
                node_feature[:, :, 1] = node_feature[:, :, 1] * (max_q - min_q + 0.0001) + min_q
                edge_feature[:, :, 1] = edge_feature[:, :, 1] * (max_velocity - min_velocity + 0.0001) + min_velocity
                node_feature = node_feature.clone().detach()
                edge_feature = edge_feature.clone().detach()
                node_updated, edge_updated, expanded_e, lambda1, lambda2, lambda3 = model(node_feature, edge_feature, D_input)


                q_feature = torch.tile(q_feature, (1, 1, 301))

                PDE_1, PDE_2 = PDE_Loss(q_feature, expanded_e, n_feature, D_feature, g, S0_feature, length_index)

                viz.heatmap(expanded_e[0, :, :, 0].cpu().detach().numpy(), win='expanded_e_capacity', opts=dict(title='expanded_e_capacity'))
                viz.heatmap(expanded_e[0, :, :, 1].cpu().detach().numpy(), win='expanded_e_velocity', opts=dict(title='expanded_e_velocity'))
                PDE_Loss_1 = Criterion_PDE(PDE_1, PDE_Label_1)
                PDE_Loss_2 = Criterion_PDE(PDE_2, PDE_Label_2)

                Loss_PDE = PDE_Loss_1 * lambda1 + PDE_Loss_2 * (1 - lambda1)
                print(PDE_Loss_2)
                print(PDE_Loss_1)
                # Backpropagate the loss
                Loss_PDE.backward()
                # Update model parameters
                Optimizer_PDE.step()
                # Clear gradients for next step
                Optimizer_PDE.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                Loss_1_all.append(PDE_Loss_1.item())
                Loss_2_all.append(PDE_Loss_2.item())
                Loss_all.append(Loss_PDE.item())
            all_output_edge = torch.zeros(test_size, num_edges, 2)
            all_output_node = torch.zeros(test_size, num_nodes)

            all_test_edge = torch.zeros(test_size, num_edges, 2)
            all_test_node = torch.zeros(test_size, num_nodes)

            time_all.append(end_time - start_time)
            if not physical_pre:
                for index, data in enumerate(devloader, 0):
                    model.eval()
                    node_feature, edge_feature, output_node_feature, output_edge_feature = data
                    node_feature = node_feature.to(device)
                    edge_feature = edge_feature.to(device)
                    output_node_feature = output_node_feature.to(device)
                    output_edge_feature = output_edge_feature.to(device)

                    node_updated, edge_updated, expanded_e, lambda1, lambda2, lambda3 = model(node_feature, edge_feature,
                                                                                              D_input)
                    all_test_edge[index * dev_size: index * dev_size + node_feature.shape[0], :,
                    :] = edge_updated.cpu().detach()
                    all_test_node[index * dev_size: index * dev_size + node_feature.shape[0],
                    :] = node_updated.cpu().detach()
                    all_output_node[index * dev_size: index * dev_size + node_feature.shape[0],
                    :] = output_node_feature.cpu().detach()
                    all_output_edge[index * dev_size: index * dev_size + node_feature.shape[0], :,
                    :] = output_edge_feature.cpu().detach()

                    all_output_edge[:, :, 0] = Area(all_output_edge[:, :, 0], D_origin_dev) * all_output_edge[:, :, 1]
                    all_test_edge[:, :, 0] = Area(all_test_edge[:, :, 0], D_origin_dev) * all_test_edge[:, :, 1]

                    all_output_node = all_output_node[:, dev_node]
                    all_output_edge = all_output_edge[:, dev_edge, :]
                    new_node_updated = all_test_node[:, dev_node]
                    new_edge_updated = all_test_edge[:, dev_edge, :]

                    nse_node = nse_loss(all_output_node, new_node_updated)
                    nse_edge_capacity = nse_loss(all_output_edge[:, :, 0], new_edge_updated[:, :, 0])
                    nse_edge_velocity = nse_loss(all_output_edge[:, :, 1], new_edge_updated[:, :, 1])

                    print(nse_node)
                    print(nse_edge_capacity)
                    print(nse_edge_velocity)
                    gc.collect()
                    torch.cuda.empty_cache()
                    viz.line([nse_node[0].cpu().detach().numpy()], [draw_time], win='NODE_2_NSE', name='Pred',
                             update='append',
                             opts=dict(title='NODE_2_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
                    viz.line([nse_node[1].cpu().detach().numpy()], [draw_time], win='NODE_5_NSE', name='Pred',
                             update='append',
                             opts=dict(title='NODE_5_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
                    viz.line([nse_node[2].cpu().detach().numpy()], [draw_time], win='NODE_8_NSE', name='Pred',
                             update='append',
                             opts=dict(title='NODE_8_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
                    viz.line([nse_node[3].cpu().detach().numpy()], [draw_time], win='NODE_11_NSE', name='Pred',
                             update='append',
                             opts=dict(title='NODE_11_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
                    viz.line([nse_node[4].cpu().detach().numpy()], [draw_time], win='NODE_14_NSE', name='Pred',
                             update='append',
                             opts=dict(title='NODE_14_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))
                    viz.line([nse_node[5].cpu().detach().numpy()], [draw_time], win='NODE_17_NSE', name='Pred',
                             update='append',
                             opts=dict(title='NODE_17_NSE', linecolor=np.array([[0, 0, 255]]), dash=np.array(['solid'])))

                    viz.line([nse_edge_capacity[0].cpu().detach().numpy()], [draw_time], win='Link_5_6_Flow_NSE',
                             name='Pred', update='append',
                             opts=dict(title='Link_5_6_Flow_NSE', linecolor=np.array([[0, 0, 255]]),
                                       dash=np.array(['solid'])))
                    viz.line([nse_edge_velocity[0].cpu().detach().numpy()], [draw_time], win='Link_5_6_Velocity_NSE',
                             name='Pred', update='append',
                             opts=dict(title='Link_5_6_Velocity_NSE', linecolor=np.array([[0, 0, 255]]),
                                       dash=np.array(['solid'])))
                    viz.line([nse_edge_capacity[1].cpu().detach().numpy()], [draw_time], win='Link_14_15_Flow_NSE',
                             name='Pred', update='append',
                             opts=dict(title='Link_14_15_Flow_NSE', linecolor=np.array([[0, 0, 255]]),
                                       dash=np.array(['solid'])))
                    viz.line([nse_edge_velocity[1].cpu().detach().numpy()], [draw_time], win='Link_14_15_Velocity_NSE',
                             name='Pred', update='append',
                             opts=dict(title='Link_14_15_Velocity_NSE', linecolor=np.array([[0, 0, 255]]),
                                       dash=np.array(['solid'])))

                    viz.line(new_node_updated[:, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_2',
                             name='Pred', update='replace', opts=dict(title='NODE_2'))
                    viz.line(all_output_node[:, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_2',
                             name='True', update='replace', opts=dict(title='NODE_2'))
                    viz.line(new_node_updated[:, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_5',
                             name='Pred', update='replace', opts=dict(title='NODE_5'))
                    viz.line(all_output_node[:, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_5',
                             name='True', update='replace', opts=dict(title='NODE_5'))
                    viz.line(new_node_updated[:, 2].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_8',
                             name='Pred', update='replace', opts=dict(title='NODE_8'))
                    viz.line(all_output_node[:, 2].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_8',
                             name='True', update='replace', opts=dict(title='NODE_8'))
                    viz.line(new_node_updated[:, 3].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_11',
                             name='Pred', update='replace', opts=dict(title='NODE_11'))
                    viz.line(all_output_node[:, 3].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_11',
                             name='True', update='replace', opts=dict(title='NODE_11'))
                    viz.line(new_node_updated[:, 4].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_14',
                             name='Pred', update='replace', opts=dict(title='NODE_14'))
                    viz.line(all_output_node[:, 4].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_14',
                             name='True', update='replace', opts=dict(title='NODE_14'))
                    viz.line(new_node_updated[:, 5].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_17',
                             name='Pred', update='replace', opts=dict(title='NODE_17'))
                    viz.line(all_output_node[:, 5].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='NODE_17',
                             name='True', update='replace', opts=dict(title='NODE_17'))

                    viz.line(new_edge_updated[:, 0, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_5_6_Flow',
                             name='Pred', update='replace', opts=dict(title='Link_5_6_Flow'))
                    viz.line(all_output_edge[:, 0, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_5_6_Flow',
                             name='True', update='replace', opts=dict(title='Link_5_6_Flow'))
                    viz.line(new_edge_updated[:, 0, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_5_6_Velocity',
                             name='Pred', update='replace', opts=dict(title='Link_5_6_Velocity'))
                    viz.line(all_output_edge[:, 0, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_5_6_Velocity',
                             name='True', update='replace', opts=dict(title='Link_5_6_Velocity'))
                    viz.line(new_edge_updated[:, 1, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_14_15_Flow',
                             name='Pred', update='replace', opts=dict(title='Link_14_15_Flow'))
                    viz.line(all_output_edge[:, 1, 0].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_14_15_Flow',
                             name='True', update='replace', opts=dict(title='Link_14_15_Flow'))
                    viz.line(new_edge_updated[:, 1, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_14_15_Velocity',
                             name='Pred', update='replace', opts=dict(title='Link_14_15_Velocity'))
                    viz.line(all_output_edge[:, 1, 1].cpu().detach().numpy(), [time for time in range(test_size)],
                             win='Link_14_15_Velocity',
                             name='True', update='replace', opts=dict(title='Link_14_15_Velocity'))
                    draw_time = draw_time + 1
        if not physical_pre:
            for index, data in enumerate(dataloader, 0):
                model.train()
                node_feature, edge_feature, output_node_feature, output_edge_feature = data
                # Select training data by train_node_index and train_edge_index

                node_feature = node_feature.to(device)
                edge_feature = edge_feature.to(device)
                output_node_feature = output_node_feature.to(device)
                output_edge_feature = output_edge_feature.to(device)

                new_node_feature = node_feature.detach()
                new_edge_feature = edge_feature.detach()
                output_node_feature = output_node_feature.detach()
                output_edge_feature = output_edge_feature.detach()

                node_updated, edge_updated, edge_updated_full, lambda1, lambda2, lambda3 = model(new_node_feature, new_edge_feature, D_input)

                output_node_feature = output_node_feature[:, train_node]
                output_edge_feature = output_edge_feature[:, train_edge, :]
                data_node_updated = node_updated[:, train_node]
                data_edge_updated = edge_updated[:, train_edge, :]
                q_feature = node_feature[:, :, 1].unsqueeze(dim=2)
                PDE_1, PDE_2 = PDE_Loss(q_feature, edge_updated_full, n_feature, D_feature, g, S0_feature, length_index)
                PDE_Loss_1 = Criterion_PDE(PDE_1, PDE_Label_1)
                PDE_Loss_2 = Criterion_PDE(PDE_2, PDE_Label_2)
                print(PDE_Loss_1)
                print(PDE_Loss_2)
                PDE_1_all.append(PDE_Loss_1.item())
                PDE_2_all.append(PDE_Loss_2.item())
                Data_1 = Criterion_DATA(data_node_updated, output_node_feature)
                Data_2 = Criterion_DATA(data_edge_updated, output_edge_feature)

                Loss_Data = Data_1 * lambda2 + Data_2 * (1 - lambda2)

                Loss_1_all.append(Data_1.item())
                Loss_2_all.append(Data_2.item())
                Loss_all.append(Loss_Data.item())
                end_time = time.time()
                time_all.append(end_time - start_time)
                print(Data_1)
                print(Data_2)
                Loss = Loss_Data
                # Backpropagate the loss
                Loss.backward()
                # Update model parameters
                Optimizer_DATA.step()
                # Clear gradients for next step
                Optimizer_DATA.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
            torch.save(model.state_dict(), 'PIGCN_' + str(i + 1) + '.pth')

    Loss_all = pd.DataFrame(Loss_all)
    Loss_1_all = pd.DataFrame(Loss_1_all)
    Loss_2_all = pd.DataFrame(Loss_2_all)
    PDE_1_all = pd.DataFrame(PDE_1_all)
    PDE_2_all = pd.DataFrame(PDE_2_all)
    Loss_all.to_csv('Data_Loss_all.csv')
    Loss_1_all.to_csv('Data_Loss_1_all.csv')
    Loss_2_all.to_csv('Data_Loss_2_all.csv')
    PDE_1_all.to_csv('PDE_Loss_1_all.csv')
    PDE_2_all.to_csv('PDE_Loss_2_all.csv')
    nse_all.to_csv('nse_all_Data.csv')
    mae_all.to_csv('mae_all_Data.csv')
    rmse_all.to_csv('rmse_all_Data.csv')
    mape_all.to_csv('mape_all_Data.csv')
    time_all = pd.DataFrame(time_all)
    time_all.to_csv('time_all_Data.csv')

    return None
# region Best_params
space_lr = [1e-5, 1e-4, 1e-3]
space_weight_decay = [1e-7, 1e-6, 1e-5]
space_edge_hop = [2, 3, 4]
space_node_hop = [2, 3, 4]
lr = [space_lr[0], space_lr[2]]
weight_decay = [space_weight_decay[0], space_weight_decay[2]]
hops_edge = space_edge_hop[1]
hops_node = space_node_hop[1]
# region Parameter import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

dev_size = batch_size
epochs = 100
train_node_index = [2, 5, 8, 11, 14, 17]
dev_node_index = [2, 5, 8, 11, 14, 17]
train_edge_index = [5, 14]
dev_edge_index = [5, 14]
length_index = [15, 15, 9, 16, 16, 11, 17, 17, 12, 17, 17, 13, 18, 18, 14, 19, 19, 16]
length_index = torch.tensor(list(map(int, length_index))).to(device)
# length_index = length_index * 100
max_length = torch.max(length_index).to(device)
# resolution = 1 / 100
resolution = 1
# Instantiate model and run forward pass
node_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.long).to(device)
edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18]],
                          dtype=torch.long).t().contiguous().to(device)

num_edges = edge_index.shape[1]
num_nodes = len(node_index)
## Pipe physical properties
D_input = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
n_input= torch.tensor([0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009])
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
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)
test_dataset = GCN_Dataset(edge_index, length_index, file_path='')
devloader = DataLoader(test_dataset, batch_size=dev_size, shuffle=False, drop_last=False, pin_memory=False)
# endregion
# endregion
model = GCN_Model(resolution=resolution, hops_node=hops_node,
                  forward_node=[64, 128, 256, 128, 64], forward_edge=[64, 128, 256, 128, 64],
                  transcov_hidden=[64, 128, 256], transcov_edge=[32, 64, 32],
                  length_index=length_index,
                  hops_edge=hops_edge, adj_matrix=adj_matrix,
                  node_index=node_index, node_matrix=node_matrix,
                  edge_matrix=edge_matrix, edge_index=edge_index).to(device)
# Train model
PIGCN_train(model=model, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            dev_size=dev_size, D_input=D_input, n_input=n_input, S0_input=S0_input,
            epochs=epochs, physical_pre=True, physical_epochs=100,
            train_node_index=train_node_index, train_edge_index=train_edge_index,
            dev_node_index=dev_node_index, dev_edge_index=dev_edge_index,
            length_index=length_index, device=device, edge_index=edge_index,
            dataloader=dataloader, devloader=devloader)

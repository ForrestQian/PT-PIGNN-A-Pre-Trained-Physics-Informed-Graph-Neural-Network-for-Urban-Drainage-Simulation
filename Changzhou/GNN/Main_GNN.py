# Import data_cache and graph_data caches
from dataset_create import dataset_create
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
from GNN_Net import GNN_Model
from Function import create_adjacency_matrices, multiply_list_with_tensor, Area, weight_init, PDE_Loss, nse_loss, setup_seed
from tqdm import tqdm
from visdom import Visdom
import pandas as pd
import time
import gc

setup_seed(42)

def model_test(model, test_loader, node_indices, edge_indices, diameter_matrix, full_node_matrix, full_edge_matrix, need_plot=False, viz=None, draw_time=1):
    model.eval()
    target_nodes_data = torch.zeros((1728, len(node_indices)))
    output_nodes_data = torch.zeros((1728, len(node_indices)))
    target_edges_data = torch.zeros((1728, len(edge_indices), 2))
    output_edges_data = torch.zeros((1728, len(edge_indices), 2))
    with torch.no_grad():
        # tqdm for progress bar
        start_index = 0
        for batch in tqdm(test_loader, desc="Testing Batches", position=0):
            end_index = start_index + batch['node_input'].shape[0]
            node_input = batch['node_input'].to(device)
            edge_input = batch['edge_input'].to(device)
            node_target = batch['node_target'].to(device)
            edge_target = batch['edge_target'].to(device)
            node_input = node_input.reshape((node_input.shape[0]), -1, node_input.shape[-1])
            edge_input = edge_input.reshape((edge_input.shape[0]), -1, edge_input.shape[-1])
            node_target = node_target.reshape((node_target.shape[0]), -1)
            edge_target = edge_target.reshape((edge_target.shape[0]), -1, edge_target.shape[-1])
            h_out, edge_out, _, _, _ = model(node_input, edge_input, diameter_matrix, full_node_matrix, full_edge_matrix)
            target_nodes_data[start_index:end_index, :] = node_target[:, node_indices].cpu()
            output_nodes_data[start_index:end_index, :] = h_out[:, node_indices].cpu()
            target_edges_data[start_index:end_index, :, :] = edge_target[:, edge_indices, :].cpu()
            output_edges_data[start_index:end_index, :, :] = edge_out[:, edge_indices, :].cpu()
            start_index = end_index
    # Compute NSE pointwise
    node_nse = []
    for i in range(len(node_indices)):
        nse = nse_loss(output_nodes_data[:, i], target_nodes_data[:, i]).item()
        node_nse.append(nse)
    flow_nse = []
    velocity_nse = []
    for j in range(len(edge_indices)):
        flow_nse_value = nse_loss(output_edges_data[:, j, 0], target_edges_data[:, j, 0]).item()
        velocity_nse_value = nse_loss(output_edges_data[:, j, 1], target_edges_data[:, j, 1]).item()
        flow_nse.append(flow_nse_value)
        velocity_nse.append(velocity_nse_value)
    if need_plot:
        if viz is not None:
            for i in range(len(node_indices)):
                viz.line(X=torch.arange(target_nodes_data.shape[0]),
                         Y=target_nodes_data[:, i],
                         win=f'Node_{node_indices[i]}',
                         name='Target',
                         opts=dict(title=f'Node_{node_indices[i]}'))
                viz.line(X=torch.arange(output_nodes_data.shape[0]),
                         Y=output_nodes_data[:, i],
                         win=f'Node_{node_indices[i]}',
                         name='Predicted',
                         update='append')
                viz.line([node_nse[i]], [draw_time], win='Node_NSE',
                         name=f'Node_{node_indices[i]}',
                         update='append', opts=dict(title='Node_NSE'))
            for j in range(len(edge_indices)):
                viz.line(X=torch.arange(target_edges_data.shape[0]),
                         Y=target_edges_data[:, j, 0],
                         win=f'Edge_{edge_indices[j]}_Flow',
                         name='Target',
                         opts=dict(title=f'Edge_{edge_indices[j]}_Flow'))
                viz.line(X=torch.arange(output_edges_data.shape[0]),
                         Y=output_edges_data[:, j, 0],
                         win=f'Edge_{edge_indices[j]}_Flow',
                         name='Predicted',
                         update='append')
                viz.line(X=torch.arange(target_edges_data.shape[0]),
                         Y=target_edges_data[:, j, 1],
                         win=f'Edge_{edge_indices[j]}_Velocity',
                         name='Target',
                         opts=dict(title=f'Edge_{edge_indices[j]}_Velocity'))
                viz.line(X=torch.arange(output_edges_data.shape[0]),
                         Y=output_edges_data[:, j, 1],
                         win=f'Edge_{edge_indices[j]}_Velocity',
                         name='Predicted',
                         update='append')
                viz.line([flow_nse[j]], [draw_time], win='Edge_Flow_NSE',
                         name=f'Edge_{edge_indices[j]}',
                         update='append', opts=dict(title='Edge_Flow_NSE'))
                viz.line([velocity_nse[j]], [draw_time], win='Edge_Velocity_NSE',
                         name=f'Edge_{edge_indices[j]}',
                        update='append', opts=dict(title='Edge_Velocity_NSE'))
            # Clear redundant GPU memory and RAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return target_nodes_data, output_nodes_data, target_edges_data, output_edges_data, node_nse, flow_nse, velocity_nse

def model_data_train(model, train_loader, test_loader, train_node_indices, train_edge_indices,
                     test_node_indices, test_edge_indices, diameter_matrix, full_node_matrix,
                     full_edge_matrix, Criterion_node, Criterion_edge, Optimizer, viz=None, epochs=1):
    model.train()
    node_columns = [f'Node_{i}' for i in test_node_indices]
    flow_columns = [f'Edge_{i}_Flow' for i in test_edge_indices]
    velocity_columns = [f'Edge_{i}_Velocity' for i in test_edge_indices]
    df_node_nse = pd.DataFrame(columns=node_columns)
    df_edge_flow_nse = pd.DataFrame(columns=flow_columns)
    df_edge_velocity_nse = pd.DataFrame(columns=velocity_columns)
    df_loss = pd.DataFrame(columns=['Epoch', 'Batch', 'Node_Loss', 'Edge_Loss', 'Total_Loss'])
    df_time = pd.DataFrame(columns=['Epoch', 'Batch', 'Time_Seconds'])
    df_lambda = pd.DataFrame(columns=['Epoch', 'Batch', 'Lambda1', 'Lambda2', 'Lambda3'])
    check_interval = 4 * 27  # Run evaluation and checkpoint every 16 training batches
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        draw_time = 1
        for batch in tqdm(train_loader, desc="Training Batches", position=0):
            if draw_time % check_interval == 1:
                target_nodes, output_nodes, target_edges, output_edges, node_nse, flow_nse, velocity_nse = model_test(model, test_loader, test_node_indices,
                                                                                    test_edge_indices, diameter_matrix,
                                                                                    full_node_matrix, full_edge_matrix,
                                                                                    need_plot=True, viz=viz, draw_time=draw_time)
                new_node_nse = pd.DataFrame([node_nse], columns=node_columns)
                new_edge_flow_nse = pd.DataFrame([flow_nse], columns=flow_columns)
                new_edge_velocity_nse = pd.DataFrame([velocity_nse], columns=velocity_columns)
                if len(df_node_nse) == 0:
                    df_node_nse = new_node_nse
                else:
                    df_node_nse = pd.concat([df_node_nse, new_node_nse], ignore_index=True)
                if len(df_edge_flow_nse) == 0:
                    df_edge_flow_nse = new_edge_flow_nse
                else:
                    df_edge_flow_nse = pd.concat([df_edge_flow_nse, new_edge_flow_nse], ignore_index=True)
                if len(df_edge_velocity_nse) == 0:
                    df_edge_velocity_nse = new_edge_velocity_nse
                else:
                    df_edge_velocity_nse = pd.concat([df_edge_velocity_nse, new_edge_velocity_nse], ignore_index=True)
                df_node_nse.to_csv('node_nse.csv', index=False)
                df_edge_flow_nse.to_csv('edge_flow_nse.csv', index=False)
                df_edge_velocity_nse.to_csv('edge_velocity_nse.csv', index=False)
                df_loss.to_csv('training_loss.csv', index=False)
                df_time.to_csv('training_time.csv', index=False)
                df_lambda.to_csv('lambda_values.csv', index=False)
                torch.save(model.state_dict(),
                           f'checkpoint_data_model_epoch{epoch + 1}_batch{draw_time - 1}.pth')
                # Clear redundant GPU memory and RAM
                del target_nodes, output_nodes, target_edges, output_edges, node_nse, flow_nse, velocity_nse
                del new_node_nse, new_edge_flow_nse, new_edge_velocity_nse
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            start_time = time.time()
            node_input = batch['node_input'].to(device)
            edge_input = batch['edge_input'].to(device)
            node_target = batch['node_target'].to(device)
            edge_target = batch['edge_target'].to(device)
            node_input = node_input.reshape((node_input.shape[0]), -1, node_input.shape[-1])
            edge_input = edge_input.reshape((edge_input.shape[0]), -1, edge_input.shape[-1])
            node_target = node_target.reshape((node_target.shape[0]), -1)
            edge_target = edge_target.reshape((edge_target.shape[0]), -1, edge_target.shape[-1])
            h_out, edge_out, lambda1, lambda2, lambda3 = model(node_input, edge_input, diameter_matrix, full_node_matrix, full_edge_matrix)
            loss_node = Criterion_node(h_out[:, train_node_indices], node_target[:, train_node_indices])
            loss_edge = Criterion_edge(edge_out[:, train_edge_indices, :], edge_target[:, train_edge_indices, :])
            loss = lambda3 * loss_node + (1 - lambda3) * loss_edge
            loss.backward()
            Optimizer.step()
            Optimizer.zero_grad()
            end_time = time.time()
            elapsed_time = end_time - start_time
            new_loss_row = pd.DataFrame({'Epoch': [epoch + 1],
                                         'Batch': [draw_time],
                                         'Node_Loss': [loss_node.item()],
                                         'Edge_Loss': [loss_edge.item()],
                                         'Total_Loss': [loss.item()]})
            new_time_row = pd.DataFrame({'Epoch': [epoch + 1],
                                         'Batch': [draw_time],
                                         'Time_Seconds': [elapsed_time]})
            new_lambda_row = pd.DataFrame({'Epoch': [epoch + 1],
                                           'Batch': [draw_time],
                                           'Lambda1': [lambda1.item()],
                                           'Lambda2': [lambda2.item()],
                                           'Lambda3': [lambda3.item()]})
            if len(df_loss) == 0:
                df_loss = new_loss_row
            else:
                df_loss = pd.concat([df_loss, new_loss_row], ignore_index=True)
            if len(df_time) == 0:
                df_time = new_time_row
            else:
                df_time = pd.concat([df_time, new_time_row], ignore_index=True)
            if len(df_lambda) == 0:
                df_lambda = new_lambda_row
            else:
                df_lambda = pd.concat([df_lambda, new_lambda_row], ignore_index=True)
            if viz is not None:
                viz.line([loss_node.item()], [draw_time], win='Data_Loss', name='Node_Loss',
                         update='append', opts=dict(title='Data_Loss'))
                viz.line([loss_edge.item()], [draw_time], win='Data_Loss', name='Edge_Loss',
                         update='append', opts=dict(title='Data_Loss'))
                viz.line([loss.item()], [draw_time], win='Data_Loss', name='Total_Data_Loss',
                         update='append', opts=dict(title='Data_Loss'))
                viz.line([lambda1.item()], [draw_time], win='Lambda_Values_Fine_Tune', name='Lambda1',
                         update='append', opts=dict(title='Lambda_Values'))
                viz.line([lambda2.item()], [draw_time], win='Lambda_Values_Fine_Tune', name='Lambda2',
                         update='append', opts=dict(title='Lambda_Values'))
                viz.line([lambda3.item()], [draw_time], win='Lambda_Values_Fine_Tune', name='Lambda3',
                         update='append', opts=dict(title='Lambda_Values'))
            draw_time += 1
            # Clear redundant GPU memory and RAM
            del node_input, edge_input, node_target, edge_target, h_out, edge_out
            del loss_node, loss_edge, loss, new_loss_row, new_time_row, new_lambda_row
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


    return model

if __name__ == "__main__":
    ################## Hyperparameter settings ##################
    resolution = 1
    hopes_node = 3
    hopes_edge = 3
    forward_node = [64, 128, 64]
    forward_edge = [64, 128, 64]
    transcov_hidden = [64, 64]
    transcov_edge = [32, 32]
    lr_data = 1e-5
    weight_decay_data = 1e-7
    lr_pde = 1e-5
    weight_decay_pde = 1e-7
    ################## Device and visualization ##################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    viz = Visdom(env='Changzhou_GNN')
    ################## Dataset preparation ##################

    (edge_index, train_data_loader, test_data_loader, sensor_node_indices, sensor_link_indices, mask_node_indices, mask_link_indices,
     link_length_list, link_diameter_list, link_roughness_list, link_offset_up_list, link_offset_down_list, node_elev_list,
     node_area_list, node_pos, max_lists, min_lists) = dataset_create(train_batch_size=8, test_batch_size=8, force_recreate=True)
    print(mask_node_indices)
    # Structural information preparation
    train_node_indices = [int(i) for i in sensor_node_indices if i not in mask_node_indices]
    print(train_node_indices)
    train_edge_indices = [int(i) for i in sensor_link_indices if i not in mask_link_indices]
    test_node_indices = [int(i) for i in sensor_node_indices]
    test_edge_indices = [int(i) for i in sensor_link_indices]
    print("DataLoaders created successfully.")
    curr_index = 0
    adj_matrix, node_matrix, edge_matrix = create_adjacency_matrices(edge_index, num_nodes=len(node_elev_list), num_edges=len(link_length_list), device=device)
    # Build expanded edge_matrix from length_list: N_edge x sum(length_list)
    full_edge_matrix = torch.zeros((len(link_length_list), sum(link_length_list))).to(device)
    current_index = 0
    for i, length in enumerate(link_length_list):
        full_edge_matrix[i, current_index:current_index + length] = 1
        current_index += length
    # Physical quantity matrix preparation
    diameter_matrix = torch.tensor(link_diameter_list, dtype=torch.float32).to(device).unsqueeze(1)  # [N_edge, 1]
    full_diameter_matrix = multiply_list_with_tensor(link_diameter_list, full_edge_matrix)  # [sum(length_list), 1]
    full_roughness_matrix = multiply_list_with_tensor(link_roughness_list, full_edge_matrix)  # [sum(length_list), 1]
    full_offset_up_matrix = multiply_list_with_tensor(link_offset_up_list, full_edge_matrix)  # [sum(length_list), 1]
    full_offset_down_matrix = multiply_list_with_tensor(link_offset_down_list, full_edge_matrix)  # [sum(length_list), 1]
    full_node_matrix = torch.matmul(node_matrix, full_edge_matrix)
    row_sums = full_edge_matrix.sum(dim=1, keepdim=True)  # [162, 1], row sums
    row_sums[row_sums == 0] = 1  # avoid division by zero
    full_edge_matrix = full_edge_matrix / row_sums  # broadcast division

    row_sums = full_node_matrix.sum(dim=1, keepdim=True)  # [N_node, 1], row sums
    row_sums[row_sums == 0] = 1  # avoid division by zero
    full_node_matrix = full_node_matrix / row_sums  # broadcast division
    print("Adjacency matrices created successfully.")
    ############# Model instantiation #############
    model = GNN_Model(resolution=resolution, hops_node=hopes_node, hops_edge=hopes_edge,
                      forward_node=forward_node, forward_edge=forward_edge, length_index=link_length_list,
                      transcov_edge=transcov_edge, transcov_hidden=transcov_hidden,
                      adj_matrix=adj_matrix, node_matrix=node_matrix, edge_matrix=edge_matrix).to(device)
    model = weight_init(model)
    print("Model instantiated successfully.")
    Criterion_node = torch.nn.MSELoss()
    Criterion_edge = torch.nn.MSELoss()
    print("Loss functions created successfully.")
    Data_Optimizer = torch.optim.Adam(model.parameters(), lr=lr_data, weight_decay=weight_decay_data)
    print("Optimizer created successfully.")

    # ############## Data fine-tuning #############
    model = model_data_train(model, train_data_loader, test_data_loader, train_node_indices, train_edge_indices,
                             test_node_indices, test_edge_indices, diameter_matrix, full_node_matrix, full_edge_matrix,
                             Criterion_node, Criterion_edge, Data_Optimizer, viz, epochs=100)
    #

    # ############## Model testing #############
    target_nodes, output_nodes, target_edges, output_edges = model_test(model, test_data_loader, test_node_indices,
                                                                        test_edge_indices, diameter_matrix, full_node_matrix, full_edge_matrix)







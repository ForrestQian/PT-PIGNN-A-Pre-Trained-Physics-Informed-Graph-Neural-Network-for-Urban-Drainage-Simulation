import pandas as pd
import torch
from Function import setup_seed, weight_init, PDE_Loss, nse_loss, Area
from PIGCN_Net import GCN_Dataset, GCN_Model
from torch.utils.data import DataLoader
import gc
import time
import os
import random

# ---------- 100 random seeds; list fixed by MASTER_LIST_SEED for reproducibility ----------
NUM_SEED_RUNS = 100
MASTER_LIST_SEED = 2026
random.seed(MASTER_LIST_SEED)
SEED_LIST = random.sample(range(1, 2 ** 31), NUM_SEED_RUNS)
# Physical pretrain convergence: total loss threshold + |Delta| of Mass/Mom scalar losses vs previous step
PHYS_TOTAL_LOSS_MAX = 1e-4
PHYS_DELTA_MAX = 1e-6

# Fine-tune convergence: total/sub-loss thresholds + |Delta| of Node/Edge scalar losses vs previous step
DATA_TOTAL_LOSS_MAX = 1e-3
DATA_DELTA_MAX = 1e-5

# Logging: verbose early physical steps then interval; one line per batch in data phase (increase PHYS_LOG_INTERVAL if needed)
PHYS_LOG_FIRST_STEPS = 20
PHYS_LOG_INTERVAL = 100
SEED_TIMEOUT_SEC = 3600


def build_refined_length_index(length_index, resolution, device):
    """Same as ResolutionTest/PIGCN_Main.py: refine pipe discretization by resolution (m)."""
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    li = torch.as_tensor(length_index, dtype=torch.float32, device=device)
    refined_length = torch.round(li / float(resolution)).to(torch.long)
    return torch.clamp(refined_length, min=2)


def build_physical_feature(batch_size, length_index_tensor, D_input, n_input, S0_input, device):
    """Same as PIGCN_Main.build_physical_feature."""
    total_length = int(length_index_tensor.sum().item())
    D_feature = torch.zeros((batch_size, total_length, 301), device=device)
    n_feature = torch.zeros((batch_size, total_length, 301), device=device)
    S0_feature = torch.zeros((batch_size, total_length, 301), device=device)
    start = 0
    for i, seg_len in enumerate(length_index_tensor.tolist()):
        end = start + int(seg_len)
        D_feature[:, start:end, :] = D_input[i]
        n_feature[:, start:end, :] = n_input[i]
        S0_feature[:, start:end, :] = S0_input[i]
        start = end
    return D_feature, n_feature, S0_feature


def random_physical_batch(batch_size, num_nodes, num_edges, device):
    """Same as PIGCN_Main.random_physical_batch ([0,1] random features, no Excel normalization)."""
    h_feature = torch.rand(batch_size, num_nodes, 1, device=device)
    q_feature = torch.rand(batch_size, num_nodes, 1, device=device)
    node_feature = torch.cat((h_feature, q_feature), dim=2)
    flow_feature = torch.rand(batch_size, num_edges, 1, device=device)
    velocity_feature = torch.rand(batch_size, num_edges, 1, device=device)
    edge_feature = torch.cat((flow_feature, velocity_feature), dim=2)
    return node_feature, edge_feature, q_feature


def PIGCN_train(length_index, dev_size, train_node_index, train_edge_index,                          ## training nodes
                dev_node_index, dev_edge_index,
                D_input, n_input, S0_input,                                                    ## physical properties
                epochs, physical_pre, physical_epochs,  # physical_epochs: max steps for physical pre-training
                model, batch_size, device, lr, weight_decay,                                 ## training hyperparameters
                edge_index, dataloader, devloader,
                num_nodes=19,
                resolution=1,
                pretrained_path='',
                physical_save_path='',
                data_pretrained_path=None,
                run_seed=0,
                output_dir=None,
                data_max_steps=None,
                deadline_ts=None):  # Global max fine-tune steps (one Optimizer.step per batch); None -> epochs * len(dataloader)
    """If output_dir is set, CSVs and relative weight paths go there. run_seed controls RNG for this stage."""

    def _out(rel_path):
        if output_dir:
            return os.path.join(output_dir, rel_path)
        return rel_path

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _check_timeout(stage_name):
        if deadline_ts is None:
            return
        now = time.time()
        if now > deadline_ts:
            raise TimeoutError(
                'seed=%d timeout: %s phase within %.1fs not finished (limit %.1fs)'
                % (run_seed, stage_name, now - (deadline_ts - SEED_TIMEOUT_SEC), SEED_TIMEOUT_SEC)
            )

    ## data
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

    setup_seed(run_seed)
    test_size = 864 - 1
    num_edges = len(length_index)
    refined_length_index = build_refined_length_index(length_index, resolution, device)
    total_refined = int(refined_length_index.sum().item())
    print(
        '[grid-check] resolution=%s m, total_points=%d, refined_length_index=%s'
        % (resolution, total_refined, refined_length_index.tolist()),
        flush=True,
    )

    D_feature, n_feature, S0_feature = build_physical_feature(
        batch_size, refined_length_index, D_input, n_input, S0_input, device
    )
    D_origin_train = torch.zeros((batch_size, len(length_index)))
    D_origin_dev = torch.zeros((test_size, len(length_index)))
    D_feature_dev = torch.zeros((dev_size, sum(length_index)))
    S0_feature_dev = torch.zeros((dev_size, sum(length_index)))
    n_feature_dev = torch.zeros((dev_size, sum(length_index)))

    for nn in range(num_edges):
        start_sum = sum(length_index[:nn])
        now_sum = sum(length_index[:nn + 1])
        D_feature_dev[:, start_sum:now_sum] = D_input[nn]
        D_origin_dev[:, nn] = D_input[nn]
        D_origin_train[:, nn] = D_input[nn]
        S0_feature_dev[:, start_sum:now_sum] = S0_input[nn]
        n_feature_dev[:, start_sum:now_sum] = n_input[nn]
    D_origin_train = D_origin_train.to(device).detach()
    S0_feature_dev = S0_feature_dev.to(device).detach()
    n_feature_dev = n_feature_dev.to(device).detach()
    D_feature_dev = D_feature_dev.to(device).detach()
    if isinstance(D_input, torch.Tensor):
        D_input = D_input.clone().detach().to(device).requires_grad_(True)
    else:
        D_input = torch.tensor(D_input, dtype=torch.float32, device=device, requires_grad=True)
    g = 9.8

    # Reset loss bias weights (same as PIGCN_Main.build_model; params on device)
    model.lambda1 = torch.nn.Parameter(torch.tensor([0.50], device=device))
    model.lambda2 = torch.nn.Parameter(torch.tensor([0.50], device=device))
    model.lambda3 = torch.nn.Parameter(torch.tensor([0.50], device=device))
    model = weight_init(model)

    def _load_checkpoint(path):
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError):
            return torch.load(path, map_location=device)

    # Load model: keep weight_init if checkpoint path is missing
    if physical_pre:
        if pretrained_path and os.path.isfile(pretrained_path):
            model.load_state_dict(_load_checkpoint(pretrained_path))
        else:
            print('Note: pretrained_path not found; starting physical pre-training from random init.')
    elif data_pretrained_path and os.path.isfile(data_pretrained_path):
        model.load_state_dict(_load_checkpoint(data_pretrained_path))
    elif pretrained_path and os.path.isfile(pretrained_path):
        model.load_state_dict(_load_checkpoint(pretrained_path))
    else:
        print('Warning: data_pretrained_path / pretrained_path not found; fine-tuning from current random init.')
    model = model.to(device)

    PDE_Label_1 = torch.zeros((batch_size, total_refined, 301), device=device)
    PDE_Label_2 = torch.zeros((batch_size, total_refined, 301), device=device)
    Criterion_PDE = torch.nn.L1Loss(reduction='mean').to(device)
    Criterion_DATA = torch.nn.MSELoss(reduction='mean').to(device)
    # Same as PIGCN_Main: AdamW with lr and weight_decay only (defaults otherwise)
    Optimizer_PDE = torch.optim.AdamW(model.parameters(), lr=lr[0], weight_decay=weight_decay[0])
    Optimizer_DATA = torch.optim.AdamW(model.parameters(), lr=lr[1], weight_decay=weight_decay[1])

    # Decrement indices by one
    train_node = [i - 1 for i in train_node_index]
    dev_node = [i - 1 for i in dev_node_index]
    train_edge = [i - 1 for i in train_edge_index]
    dev_edge = [i - 1 for i in dev_edge_index]

    nse_cols = list(nse_all.columns)
    nse_after_physical = pd.DataFrame(columns=nse_cols + ['conv_time_sec'])
    nse_after_data = pd.DataFrame(columns=nse_cols + ['conv_time_sec'])
    physical_conv_times = []
    data_conv_times = []

    def eval_test_nse_row():
        """Compute per-monitor NSE on test set; return one row without time column."""
        _check_timeout('eval_test_nse_row')
        all_output_edge = torch.zeros(test_size, num_edges, 2)
        all_output_node = torch.zeros(test_size, num_nodes)
        all_test_edge = torch.zeros(test_size, num_edges, 2)
        all_test_node = torch.zeros(test_size, num_nodes)
        for index, data in enumerate(devloader, 0):
            _check_timeout('eval_test_nse_row')
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
        all_output_node = all_output_node[:, dev_node]
        all_output_edge = all_output_edge[:, dev_edge, :]
        new_node_updated = all_test_node[:, dev_node]
        new_edge_updated = all_test_edge[:, dev_edge, :]
        nse_node = nse_loss(all_output_node, new_node_updated)
        nse_edge_capacity = nse_loss(all_output_edge[:, :, 0], new_edge_updated[:, :, 0])
        nse_edge_velocity = nse_loss(all_output_edge[:, :, 1], new_edge_updated[:, :, 1])
        return [nse_node[0].item(), nse_node[1].item(), nse_node[2].item(), nse_node[3].item(), nse_node[4].item(), nse_node[5].item(),
                nse_edge_capacity[0].item(), nse_edge_velocity[0].item(), nse_edge_capacity[1].item(), nse_edge_velocity[1].item()]

    if physical_pre:
        os.makedirs(os.path.dirname(physical_save_path) or '.', exist_ok=True)
        # try:
        #     _viz_phys = Visdom()
        # except Exception:
        #     _viz_phys = None
        phys_t0 = time.time()
        phys_converged = False
        index_physical = 0
        prev_l1 = None
        prev_l2 = None
        prev_total_loss = None
        print('[physical pre-training] start | criteria: Loss_PDE<%.0e, |Delta Mass|/|Delta Mom| (adjacent-step scalar loss diff)<%.0e | at most %d steps' % (
            PHYS_TOTAL_LOSS_MAX, PHYS_DELTA_MAX, physical_epochs), flush=True)
        while index_physical < physical_epochs and not phys_converged:
            _check_timeout('physical pre-training')
            model.train()
            node_feature, edge_feature, q_feature = random_physical_batch(
                batch_size, num_nodes, num_edges, device
            )
            node_updated, edge_updated, expanded_e, lambda1, lambda2, lambda3 = model(node_feature, edge_feature, D_input)
            q_feature_expand = torch.tile(q_feature, (1, 1, expanded_e.shape[2]))
            PDE_1, PDE_2 = PDE_Loss(
                q_feature_expand, expanded_e, n_feature, D_feature, g, S0_feature,
                refined_length_index, resolution,
            )
            PDE_Loss_1 = Criterion_PDE(PDE_1, PDE_Label_1)
            PDE_Loss_2 = Criterion_PDE(PDE_2, PDE_Label_2)
            Loss_PDE = PDE_Loss_1 * lambda1 + PDE_Loss_2 * (1 - lambda1)
            Optimizer_PDE.zero_grad(set_to_none=True)
            Loss_PDE.backward()
            Optimizer_PDE.step()
            l1 = float(PDE_Loss_1.item())
            l2 = float(PDE_Loss_2.item())
            loss_val = float(Loss_PDE.item())
            mass_loss_gradient = (l1 - prev_l1) if prev_l1 is not None else None
            pde_loss_gradient = (l2 - prev_l2) if prev_l2 is not None else None
            loss_delta = abs(loss_val - prev_total_loss) if prev_total_loss is not None else None
            loss_gradient = (loss_val - prev_total_loss) if prev_total_loss is not None else None
            prev_l1 = l1
            prev_l2 = l2
            prev_total_loss = loss_val
            Loss_1_all.append(l1)
            Loss_2_all.append(l2)
            Loss_all.append(loss_val)
            phys_converged = (
                mass_loss_gradient is not None
                and pde_loss_gradient is not None
                and loss_val < PHYS_TOTAL_LOSS_MAX
                and abs(mass_loss_gradient) < PHYS_DELTA_MAX
                and abs(pde_loss_gradient) < PHYS_DELTA_MAX
            )
            index_physical += 1
            lam1 = float(lambda1.detach().item()) if hasattr(lambda1, 'detach') else float(lambda1)
            do_log = (
                index_physical <= PHYS_LOG_FIRST_STEPS
                or index_physical % PHYS_LOG_INTERVAL == 0
                or phys_converged
            )
            if do_log:
                print(
                    '[res=%s] iter=%05d mass_loss=%.8e pde_loss=%.8e total_loss=%.8e delta_loss=%.8e '
                    'loss_gradient=%.8e mass_loss_gradient=%.8e pde_loss_gradient=%.8e λ1=%.4f converged=%s'
                    % (
                        resolution,
                        index_physical,
                        l1,
                        l2,
                        loss_val,
                        loss_delta if loss_delta is not None else float('nan'),
                        loss_gradient if loss_gradient is not None else float('nan'),
                        mass_loss_gradient if mass_loss_gradient is not None else float('nan'),
                        pde_loss_gradient if pde_loss_gradient is not None else float('nan'),
                        lam1,
                        phys_converged,
                    ),
                    flush=True,
                )
            if phys_converged:
                print(
                    '[res=%s] converged at iter=%05d: total_loss=%.8e < loss_threshold=%.8e, '
                    '|mass_grad|=%.8e, |pde_grad|=%.8e < delta_threshold=%.8e'
                    % (
                        resolution,
                        index_physical,
                        loss_val,
                        PHYS_TOTAL_LOSS_MAX,
                        abs(mass_loss_gradient),
                        abs(pde_loss_gradient),
                        PHYS_DELTA_MAX,
                    ),
                    flush=True,
                )
                break
        physical_conv_sec = time.time() - phys_t0
        physical_conv_times.append(physical_conv_sec)
        torch.save(model.state_dict(), physical_save_path)
        if output_dir:
            print('  Physical weights saved.', flush=True)
        nse_row = eval_test_nse_row()
        nse_after_physical.loc[0] = nse_row + [physical_conv_sec]
        print(f'Physical pre-training finished: converged={phys_converged}, steps={index_physical}, elapsed={physical_conv_sec:.2f}s, NSE row={nse_row}')
        if not phys_converged:
            print('Warning: Physical pre-training did not meet criteria within physical_epochs; saved weights and logged test NSE')
    else:
        data_converged = False
        data_t0 = time.time()
        batch_step = 0
        prev_mean_d1 = None
        prev_mean_d2 = None
        n_batches = len(dataloader)
        if data_max_steps is None:
            data_max_steps = int(epochs) * int(n_batches)
        print('[data fine-tuning] start | criteria: Mean Loss_node/Loss_edge over all batches in each epoch;'
              'mean <%.0e and vs previous full epoch |Delta node|/|Delta edge| (mean difference)<%.0e | per epoch %d  batches | at most %d steps (global optimizer steps)'
              % (DATA_TOTAL_LOSS_MAX, DATA_DELTA_MAX, n_batches, data_max_steps), flush=True)
        epoch = 0
        while not data_converged and batch_step < data_max_steps:
            _check_timeout('data fine-tuning')
            print('========== Epoch %d (at most %d steps) ==========' % (epoch, data_max_steps), flush=True)

            all_output_edge = torch.zeros(test_size, num_edges, 2)
            all_output_node = torch.zeros(test_size, num_nodes)
            all_test_edge = torch.zeros(test_size, num_edges, 2)
            all_test_node = torch.zeros(test_size, num_nodes)

            for index, data in enumerate(devloader, 0):
                _check_timeout('data fine-tuning-validation')
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
            nse_all.loc[epoch] = [nse_node[0].item(), nse_node[1].item(), nse_node[2].item(), nse_node[3].item(), nse_node[4].item(), nse_node[5].item(),
                                nse_edge_capacity[0].item(), nse_edge_velocity[0].item(), nse_edge_capacity[1].item(), nse_edge_velocity[1].item()]
            mae_all.loc[epoch] = [mae_node[0].item(), mae_node[1].item(), mae_node[2].item(), mae_node[3].item(), mae_node[4].item(), mae_node[5].item(),
                                mae_edge_capacity[0].item(), mae_edge_velocity[0].item(), mae_edge_capacity[1].item(), mae_edge_velocity[1].item()]
            rmse_all.loc[epoch] = [rmse_node[0].item(), rmse_node[1].item(), rmse_node[2].item(), rmse_node[3].item(), rmse_node[4].item(), rmse_node[5].item(),
                                mae_edge_capacity[0].item(), mae_edge_velocity[0].item(), mae_edge_capacity[1].item(), mae_edge_velocity[1].item()]
            mape_all.loc[epoch] = [mape_node[0].item(), mape_node[1].item(), mape_node[2].item(), mape_node[3].item(), mape_node[4].item(), mape_node[5].item(),
                                mae_edge_capacity[0].item(), mae_edge_velocity[0].item(), mae_edge_capacity[1].item(), mae_edge_velocity[1].item()]
            print('  [Before this epoch - test set] NSE node[6]: %s'
                  % ', '.join(['%.4f' % nse_node[j].item() for j in range(6)]), flush=True)
            print('  [Before this epoch - test set] NSE edge capacity[2] vel[2]: %s, %s'
                  % (', '.join(['%.4f' % nse_edge_capacity[j].item() for j in range(2)]),
                     ', '.join(['%.4f' % nse_edge_velocity[j].item() for j in range(2)])), flush=True)
            gc.collect()
            torch.cuda.empty_cache()

            sum_d1 = 0.0
            sum_d2 = 0.0
            sum_ld = 0.0
            batches_trained = 0
            for index, data in enumerate(dataloader, 0):
                _check_timeout('data fine-tuning-training')
                if batch_step >= data_max_steps:
                    break
                model.train()
                node_feature, edge_feature, output_node_feature, output_edge_feature = data
                node_feature = node_feature.to(device)
                edge_feature = edge_feature.to(device)
                output_node_feature = output_node_feature.to(device)
                output_edge_feature = output_edge_feature.to(device)
                new_node_feature = node_feature.detach()
                new_edge_feature = edge_feature.detach()
                output_node_feature = output_node_feature.detach()
                output_edge_feature = output_edge_feature.detach()
                Optimizer_DATA.zero_grad(set_to_none=True)
                node_updated, edge_updated, edge_updated_full, lambda1, lambda2, lambda3 = model(new_node_feature, new_edge_feature, D_input)
                output_node_feature_sub = output_node_feature[:, train_node]
                output_edge_feature_sub = output_edge_feature[:, train_edge, :]
                data_node_updated = node_updated[:, train_node]
                data_edge_updated = edge_updated[:, train_edge, :]
                q_feature = node_feature[:, :, 1].unsqueeze(dim=2)
                q_feature_expand = torch.tile(q_feature, (1, 1, edge_updated_full.shape[2]))
                PDE_1, PDE_2 = PDE_Loss(
                    q_feature_expand, edge_updated_full, n_feature, D_feature, g, S0_feature,
                    refined_length_index, resolution,
                )
                PDE_Loss_1 = Criterion_PDE(PDE_1, PDE_Label_1)
                PDE_Loss_2 = Criterion_PDE(PDE_2, PDE_Label_2)
                pde1 = float(PDE_Loss_1.item())
                pde2 = float(PDE_Loss_2.item())
                PDE_1_all.append(pde1)
                PDE_2_all.append(pde2)
                Data_1 = Criterion_DATA(data_node_updated, output_node_feature_sub)
                Data_2 = Criterion_DATA(data_edge_updated, output_edge_feature_sub)
                Loss_Data = Data_1 * lambda2 + Data_2 * (1 - lambda2)
                Loss_Data.backward()
                Optimizer_DATA.step()
                d1 = float(Data_1.item())
                d2 = float(Data_2.item())
                ld = float(Loss_Data.item())
                sum_d1 += d1
                sum_d2 += d2
                sum_ld += ld
                batches_trained += 1
                Loss_1_all.append(d1)
                Loss_2_all.append(d2)
                Loss_all.append(ld)
                batch_step += 1
                time_all.append(time.time() - start_time)
                lam2 = float(lambda2.detach().item()) if hasattr(lambda2, 'detach') else float(lambda2)
                elapsed = time.time() - data_t0
                print(
                    '  [data] ep %d batch %d/%d (global step %d) | Loss_node=%.6e Loss_edge=%.6e Loss_total=%.6e | '
                    'PDE_Mass=%.6e PDE_Mom=%.6e | λ2=%.4f | elapsed %.1fs'
                    % (epoch, index + 1, n_batches, batch_step, d1, d2, ld, pde1, pde2, lam2, elapsed),
                    flush=True,
                )
                if batch_step >= data_max_steps:
                    print('Fine-tuning: reached global max steps data_max_steps=%d, stop training' % data_max_steps, flush=True)
                    break
                gc.collect()
                torch.cuda.empty_cache()

            if batches_trained == 0:
                break

            mean_d1 = sum_d1 / batches_trained
            mean_d2 = sum_d2 / batches_trained
            mean_ld = sum_ld / batches_trained
            full_epoch = batches_trained == n_batches
            delta_mean_d1 = (mean_d1 - prev_mean_d1) if prev_mean_d1 is not None else None
            delta_mean_d2 = (mean_d2 - prev_mean_d2) if prev_mean_d2 is not None else None
            print(
                '  [This epoch mean training loss] batches=%d/%s | mean_node=%.6e mean_edge=%.6e mean_total=%.6e | '
                'full epoch=%s | vs previous epoch mean diff |Delta node|=%s |Delta edge|=%s'
                % (
                    batches_trained,
                    n_batches,
                    mean_d1,
                    mean_d2,
                    mean_ld,
                    full_epoch,
                    '%.6e' % abs(delta_mean_d1) if delta_mean_d1 is not None else 'n/a',
                    '%.6e' % abs(delta_mean_d2) if delta_mean_d2 is not None else 'n/a',
                ),
                flush=True,
            )
            if full_epoch and prev_mean_d1 is not None and prev_mean_d2 is not None:
                data_converged = (
                    mean_ld < DATA_TOTAL_LOSS_MAX
                    and mean_d1 < DATA_TOTAL_LOSS_MAX
                    and mean_d2 < DATA_TOTAL_LOSS_MAX
                    and abs(delta_mean_d1) < DATA_DELTA_MAX
                    and abs(delta_mean_d2) < DATA_DELTA_MAX
                )
                if data_converged:
                    data_conv_sec = time.time() - data_t0
                    data_conv_times.append(data_conv_sec)
                    nse_row = eval_test_nse_row()
                    nse_after_data.loc[0] = nse_row + [data_conv_sec]
                    print(
                        'Fine-tuning converged: epoch=%d ended; epoch batch means meet criteria, batch_step=%d, elapsed=%.2fs, NSE=%s'
                        % (epoch, batch_step, data_conv_sec, nse_row),
                        flush=True,
                    )
            if full_epoch:
                prev_mean_d1 = mean_d1
                prev_mean_d2 = mean_d2

            epoch += 1
            if data_converged:
                break

        if not data_converged:
            print('Warning: Fine-tuning did not converge within data_max_steps=%d; saving model and logging test NSE' % data_max_steps)
            data_conv_sec = time.time() - data_t0
            data_conv_times.append(data_conv_sec)
            nse_row = eval_test_nse_row()
            nse_after_data.loc[0] = nse_row + [data_conv_sec]

        _data_ckpt = _out('PIGCN_data_converged.pth')
        torch.save(model.state_dict(), _data_ckpt)
        if output_dir:
            print('  Fine-tuning weights saved.', flush=True)

    Loss_all = pd.DataFrame(Loss_all)
    Loss_1_all = pd.DataFrame(Loss_1_all)
    Loss_2_all = pd.DataFrame(Loss_2_all)
    PDE_1_all = pd.DataFrame(PDE_1_all)
    PDE_2_all = pd.DataFrame(PDE_2_all)
    Loss_all.to_csv(_out('Data_Loss_all.csv'))
    Loss_1_all.to_csv(_out('Data_Loss_1_all.csv'))
    Loss_2_all.to_csv(_out('Data_Loss_2_all.csv'))
    PDE_1_all.to_csv(_out('PDE_Loss_1_all.csv'))
    PDE_2_all.to_csv(_out('PDE_Loss_2_all.csv'))
    nse_all.to_csv(_out('nse_all_Data.csv'))
    mae_all.to_csv(_out('mae_all_Data.csv'))
    rmse_all.to_csv(_out('rmse_all_Data.csv'))
    mape_all.to_csv(_out('mape_all_Data.csv'))
    time_all = pd.DataFrame(time_all)
    time_all.to_csv(_out('time_all_Data.csv'))
    if physical_pre:
        pd.DataFrame({'physical_conv_time_sec': physical_conv_times}).to_csv(_out('physical_conv_time.csv'), index=False)
        nse_after_physical.to_csv(_out('nse_test_after_physical.csv'), index=False)
    else:
        pd.DataFrame({'data_conv_time_sec': data_conv_times}).to_csv(_out('data_conv_time.csv'), index=False)
        nse_after_data.to_csv(_out('nse_test_after_data.csv'), index=False)

    ret = {
        'run_seed': run_seed,
        'physical_pre': physical_pre,
        'physical_conv_time_sec': float(physical_conv_times[0]) if physical_pre and physical_conv_times else None,
        'data_conv_time_sec': float(data_conv_times[0]) if (not physical_pre) and data_conv_times else None,
        'nse_after_physical': nse_after_physical if physical_pre and not nse_after_physical.empty else None,
        'nse_after_data': nse_after_data if (not physical_pre) and not nse_after_data.empty else None,
        'output_dir': output_dir,
    }
    return ret
# region Best_params
space_lr = [1e-5, 1e-4, 1e-3]
space_weight_decay = [1e-7, 1e-6, 1e-5]
space_edge_hop = [2, 3, 4]
space_node_hop = [2, 3, 4]
lr = [space_lr[2], space_lr[2]]
weight_decay = [space_weight_decay[2], space_weight_decay[2]]
hops_edge = space_edge_hop[1]
hops_node = space_node_hop[1]
# region Parameter import
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8

dev_size = batch_size
epochs = 100
# Fine-tune: global max optimizer steps (1 per batch); same counting as physical_epochs; None -> epochs * len(dataloader)
DATA_MAX_STEPS = 500000
train_node_index = [2, 5, 8, 11, 14, 17]
dev_node_index = [2, 5, 8, 11, 14, 17]
train_edge_index = [5, 14]
dev_edge_index = [5, 14]
length_index = [15, 15, 9, 16, 16, 11, 17, 17, 12, 17, 17, 13, 18, 18, 14, 19, 19, 16]
length_index = torch.tensor(list(map(int, length_index))).to(device)
# length_index = length_index * 100
max_length = torch.max(length_index).to(device)
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
def _build_model():
    return GCN_Model(resolution=resolution, hops_node=hops_node,
                     forward_node=[64, 128, 256, 128, 64], forward_edge=[64, 128, 256, 128, 64],
                     transcov_hidden=[64, 128, 256], transcov_edge=[32, 64, 32],
                     length_index=length_index,
                     hops_edge=hops_edge, adj_matrix=adj_matrix,
                     node_index=node_index, node_matrix=node_matrix,
                     edge_matrix=edge_matrix, edge_index=edge_index).to(device)




RESULT_ROOT = ''
os.makedirs(RESULT_ROOT, exist_ok=True)
pd.DataFrame({'run_index': range(NUM_SEED_RUNS), 'seed': SEED_LIST}).to_csv(
    os.path.join(RESULT_ROOT, 'seed_list_generated.csv'), index=False)
print('Generated %d distinct random seeds.' % NUM_SEED_RUNS, flush=True)

all_summary_rows = []

for run_idx, seed in enumerate(SEED_LIST):
    run_dir = os.path.join(RESULT_ROOT, 'seed_%d' % seed)
    os.makedirs(run_dir, exist_ok=True)
    phys_ckpt = os.path.join(run_dir, 'PIGCN_phys_converged.pth')
    per_seed_file = os.path.join(run_dir, 'results_seed_%d.csv' % seed)
    seed_deadline = time.time() + SEED_TIMEOUT_SEC
    print('\n' + '=' * 70, flush=True)
    print('Run %d / %d | seed = %d' % (run_idx + 1, NUM_SEED_RUNS, seed), flush=True)
    print('=' * 70 + '\n', flush=True)
    if os.path.isfile(per_seed_file):
        print('seed=%d: result file exists, skipping training.' % seed, flush=True)
        try:
            existed_df = pd.read_csv(per_seed_file)
            if not existed_df.empty:
                all_summary_rows.append(existed_df.iloc[0].to_dict())
        except Exception as e:
            print('Failed to read existing results (skip training only, continue other seeds): %s' % e, flush=True)
        continue
    try:
        print('---------- Phase 1: physical pre-training (seed=%d) ----------' % seed, flush=True)
        r1 = PIGCN_train(
            model=_build_model(), lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            dev_size=dev_size, D_input=D_input, n_input=n_input, S0_input=S0_input,
            epochs=epochs, physical_pre=True, physical_epochs=500000,
            train_node_index=train_node_index, train_edge_index=train_edge_index,
            dev_node_index=dev_node_index, dev_edge_index=dev_edge_index,
            length_index=length_index, device=device, edge_index=edge_index,
            dataloader=dataloader, devloader=devloader,
            num_nodes=num_nodes, resolution=resolution,
            pretrained_path='',
            physical_save_path=phys_ckpt,
            data_pretrained_path=None,
            run_seed=seed,
            output_dir=run_dir,
            deadline_ts=seed_deadline,
        )
        if time.time() > seed_deadline:
            raise TimeoutError('seed=%d exceeded after phase 1 (%.1fs), skipping phase 2' % (seed, SEED_TIMEOUT_SEC))
        print('---------- Phase 2: data fine-tuning (seed=%d) ----------' % seed, flush=True)
        r2 = PIGCN_train(
            model=_build_model(), lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            dev_size=dev_size, D_input=D_input, n_input=n_input, S0_input=S0_input,
            epochs=epochs, physical_pre=False, physical_epochs=500000,
            data_max_steps=DATA_MAX_STEPS,
            train_node_index=train_node_index, train_edge_index=train_edge_index,
            dev_node_index=dev_node_index, dev_edge_index=dev_edge_index,
            length_index=length_index, device=device, edge_index=edge_index,
            dataloader=dataloader, devloader=devloader,
            num_nodes=num_nodes, resolution=resolution,
            pretrained_path='',
            physical_save_path=phys_ckpt,
            data_pretrained_path=phys_ckpt,
            run_seed=seed,
            output_dir=run_dir,
            deadline_ts=seed_deadline,
        )
        row = {
            'run_index': run_idx,
            'seed': seed,
            'physical_conv_time_sec': r1['physical_conv_time_sec'],
            'data_conv_time_sec': r2['data_conv_time_sec'],
        }
        if r1['nse_after_physical'] is not None and not r1['nse_after_physical'].empty:
            for c in r1['nse_after_physical'].columns:
                row['phys_' + c] = r1['nse_after_physical'].iloc[0][c]
        if r2['nse_after_data'] is not None and not r2['nse_after_data'].empty:
            for c in r2['nse_after_data'].columns:
                if c == 'conv_time_sec':
                    continue
                row['test_' + c] = r2['nse_after_data'].iloc[0][c]
        pd.DataFrame([row]).to_csv(per_seed_file, index=False)
        all_summary_rows.append(row)
        print('Results for this seed saved.', flush=True)
    except TimeoutError as e:
        print('seed=%d Skipped due to timeout: %s' % (seed, e), flush=True)
        err_row = {'run_index': run_idx, 'seed': seed, 'error': str(e), 'timeout_sec': SEED_TIMEOUT_SEC}
        pd.DataFrame([err_row]).to_csv(os.path.join(run_dir, 'results_seed_%d_TIMEOUT.csv' % seed), index=False)
        all_summary_rows.append(err_row)
    except Exception as e:
        print('seed=%d Run failed: %s' % (seed, e), flush=True)
        err_row = {'run_index': run_idx, 'seed': seed, 'error': str(e)}
        pd.DataFrame([err_row]).to_csv(os.path.join(run_dir, 'results_seed_%d_ERROR.csv' % seed), index=False)
        all_summary_rows.append(err_row)

if all_summary_rows:
    pd.DataFrame(all_summary_rows).to_csv(os.path.join(RESULT_ROOT, 'all_seeds_summary.csv'), index=False)
    print('\nAll done: summary table written.', flush=True)

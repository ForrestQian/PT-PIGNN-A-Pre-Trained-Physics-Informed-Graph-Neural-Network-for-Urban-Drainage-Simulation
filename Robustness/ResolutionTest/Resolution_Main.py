import time
from pathlib import Path

import pandas as pd
import torch

from Function import PDE_Loss, setup_seed, weight_init
from PIGCN_Net import GCN_Model


# ========== PyCharm run configuration (edit as needed) ==========
# 5m, 2m, 1m, 5dm, 2dm, 1dm, 5cm
RESOLUTION = 5
BATCH_SIZE = 8
MAX_ITERS = 10000
CONVERGE_LOSS_THRESHOLD = 1e-5
CONVERGE_DELTA_THRESHOLD = 1e-4
STOP_WHEN_CONVERGED = True
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
OUTPUT_DIR = ''


def build_graph(device):
    length_index = torch.tensor(
        [15, 15, 9, 16, 16, 11, 17, 17, 12, 17, 17, 13, 18, 18, 14, 19, 19, 16],
        dtype=torch.long,
        device=device,
    )
    node_index = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        dtype=torch.long,
        device=device,
    )
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [17, 18],
        ],
        dtype=torch.long,
        device=device,
    ).t().contiguous()
    num_nodes = len(node_index)
    num_edges = edge_index.shape[1]

    adj_matrix = torch.zeros(num_edges, num_nodes, device=device)
    node_matrix = torch.zeros(num_nodes, num_nodes, device=device)
    edge_matrix = torch.zeros(num_edges, num_edges, device=device)

    for i in range(num_edges):
        adj_matrix[i, edge_index[0, i]] = 1
        adj_matrix[i, edge_index[1, i]] = 1

    node_matrix[edge_index[0], edge_index[1]] = 1
    node_matrix[edge_index[1], edge_index[0]] = 1
    node_matrix[edge_index[0], edge_index[0]] = 1
    node_matrix[edge_index[1], edge_index[1]] = 1

    for i in range(num_edges):
        for j in range(i, num_edges):
            start_i, end_i = edge_index[:, i]
            start_j, end_j = edge_index[:, j]
            if start_i == start_j or start_i == end_j or end_i == start_j or end_i == end_j:
                edge_matrix[i, j] = 1
                edge_matrix[j, i] = 1

    return length_index, node_index, edge_index, adj_matrix, node_matrix, edge_matrix


def build_refined_length_index(length_index, resolution):
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    refined_length = torch.round(length_index.to(torch.float32) / float(resolution)).to(torch.long)
    return torch.clamp(refined_length, min=2)


def build_physical_feature(batch_size, length_index, D_input, n_input, S0_input, device):
    total_length = int(length_index.sum().item())
    D_feature = torch.zeros((batch_size, total_length, 301), device=device)
    n_feature = torch.zeros((batch_size, total_length, 301), device=device)
    S0_feature = torch.zeros((batch_size, total_length, 301), device=device)
    start = 0
    for i, seg_len in enumerate(length_index.tolist()):
        end = start + seg_len
        D_feature[:, start:end, :] = D_input[i]
        n_feature[:, start:end, :] = n_input[i]
        S0_feature[:, start:end, :] = S0_input[i]
        start = end
    return D_feature, n_feature, S0_feature


def build_model(resolution, length_index, node_index, edge_index, adj_matrix, node_matrix, edge_matrix, device):
    model = GCN_Model(
        resolution=resolution,
        hops_node=3,
        hops_edge=3,
        forward_node=[64, 128, 256, 128, 64],
        forward_edge=[64, 128, 256, 128, 64],
        transcov_hidden=[64, 128, 256],
        transcov_edge=[32, 64, 32],
        length_index=length_index,
        adj_matrix=adj_matrix,
        node_index=node_index,
        node_matrix=node_matrix,
        edge_matrix=edge_matrix,
        edge_index=edge_index,
    ).to(device)
    model.lambda1 = torch.nn.Parameter(torch.tensor([0.50], device=device))
    model.lambda2 = torch.nn.Parameter(torch.tensor([0.50], device=device))
    model.lambda3 = torch.nn.Parameter(torch.tensor([0.50], device=device))
    model = weight_init(model)
    return model


def random_physical_batch(batch_size, num_nodes, num_edges, D_input, device):
    h_feature = torch.rand(batch_size, num_nodes, 1, device=device)
    q_feature = torch.rand(batch_size, num_nodes, 1, device=device)
    node_feature = torch.cat((h_feature, q_feature), dim=2)

    flow_feature = torch.rand(batch_size, num_edges, 1, device=device)
    velocity_feature = torch.rand(batch_size, num_edges, 1, device=device)
    edge_feature = torch.cat((flow_feature, velocity_feature), dim=2)
    return node_feature, edge_feature, q_feature


def gpu_memory_mb(value):
    return float(value) / 1024.0 / 1024.0


def profile_one_step(model, optimizer, node_feature, edge_feature, q_feature, D_input, length_index, D_feature, n_feature, S0_feature, pde_label_1, pde_label_2, criterion, g, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    node_updated, edge_updated, expanded_e, lambda1, _, _ = model(node_feature, edge_feature, D_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    forward_peak_mem_mb = gpu_memory_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else -1.0

    q_feature_expand = torch.tile(q_feature, (1, 1, expanded_e.shape[2]))
    pde_1, pde_2 = PDE_Loss(q_feature_expand, expanded_e, n_feature, D_feature, g, S0_feature, length_index, RESOLUTION)
    mass_loss = criterion(pde_1, pde_label_1)
    pde_loss = criterion(pde_2, pde_label_2)
    total_loss = mass_loss * lambda1 + pde_loss * (1 - lambda1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t2 = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t3 = time.perf_counter()

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_peak_mem_mb = gpu_memory_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else -1.0

    profile_row = {
        "forward_time_s": t1 - t0,
        "loss_compute_time_s": t2 - t1,
        "backward_and_step_time_s": t3 - t2,
        "mass_loss": float(mass_loss.item()),
        "pde_loss": float(pde_loss.item()),
        "total_loss": float(total_loss.item()),
        "forward_peak_memory_allocated_mb": forward_peak_mem_mb,
        "train_peak_memory_allocated_mb": train_peak_mem_mb,
        "model_params_total": int(params_total),
        "model_params_trainable": int(params_trainable),
    }
    return profile_row


def physical_pretrain_for_resolution(resolution, config, device, output_dir):
    length_index, node_index, edge_index, adj_matrix, node_matrix, edge_matrix = build_graph(device)
    model = build_model(
        resolution=resolution,
        length_index=length_index,
        node_index=node_index,
        edge_index=edge_index,
        adj_matrix=adj_matrix,
        node_matrix=node_matrix,
        edge_matrix=edge_matrix,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = torch.nn.L1Loss(reduction="mean").to(device)
    num_edges = edge_index.shape[1]
    num_nodes = len(node_index)

    D_input = torch.tensor(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        dtype=torch.float32,
        device=device,
    )
    n_input = torch.tensor([0.009] * num_edges, dtype=torch.float32, device=device)
    s0_input = torch.tensor([0.005] * num_edges, dtype=torch.float32, device=device)
    refined_length_index = build_refined_length_index(length_index, resolution)
    print(
        f"[grid-check] resolution={resolution}m, "
        f"total_points={int(refined_length_index.sum().item())}, "
        f"example:15m->{int(torch.round(torch.tensor(15.0) / float(resolution)).item())} points"
    )
    print(f"[grid-check] raw_length_index={length_index.tolist()}")
    print(f"[grid-check] refined_length_index={refined_length_index.tolist()}")
    D_feature, n_feature, S0_feature = build_physical_feature(
        config["batch_size"], refined_length_index, D_input, n_input, s0_input, device
    )
    pde_label_1 = torch.zeros((config["batch_size"], int(refined_length_index.sum().item()), 301), device=device)
    pde_label_2 = torch.zeros((config["batch_size"], int(refined_length_index.sum().item()), 301), device=device)
    g = 9.8

    node_feature, edge_feature, q_feature = random_physical_batch(
        config["batch_size"], num_nodes, num_edges, D_input, device
    )
    profile_row = profile_one_step(
        model,
        optimizer,
        node_feature,
        edge_feature,
        q_feature,
        D_input,
        refined_length_index,
        D_feature,
        n_feature,
        S0_feature,
        pde_label_1,
        pde_label_2,
        criterion,
        g,
        device,
    )
    profile_row["resolution"] = resolution

    train_rows = []
    converged_iter = -1
    prev_total_loss = None
    prev_mass_loss = None
    prev_pde_loss = None
    start_train = time.perf_counter()
    for it in range(1, config["max_iters"] + 1):
        iter_start = time.perf_counter()
        model.train()
        node_feature, edge_feature, q_feature = random_physical_batch(
            config["batch_size"], num_nodes, num_edges, D_input, device
        )
        _, _, expanded_e, lambda1, _, _ = model(node_feature, edge_feature, D_input)
        q_feature_expand = torch.tile(q_feature, (1, 1, expanded_e.shape[2]))
        pde_1, pde_2 = PDE_Loss(q_feature_expand, expanded_e, n_feature, D_feature, g, S0_feature, refined_length_index, RESOLUTION)
        mass_loss = criterion(pde_1, pde_label_1)
        pde_loss = criterion(pde_2, pde_label_2)
        total_loss = mass_loss * lambda1 + pde_loss * (1 - lambda1)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        iter_end = time.perf_counter()
        elapsed = iter_end - start_train
        train_rows.append(
            {
                "resolution": resolution,
                "iter": it,
                "mass_loss": float(mass_loss.item()),
                "pde_loss": float(pde_loss.item()),
                "total_loss": float(total_loss.item()),
                "iter_time_s": iter_end - iter_start,
                "elapsed_time_s": elapsed,
            }
        )

        current_mass_loss = float(mass_loss.item())
        current_pde_loss = float(pde_loss.item())
        current_total_loss = float(total_loss.item())
        loss_delta = abs(current_total_loss - prev_total_loss) if prev_total_loss is not None else None
        loss_gradient = (current_total_loss - prev_total_loss) if prev_total_loss is not None else None
        mass_loss_gradient = (current_mass_loss - prev_mass_loss) if prev_mass_loss is not None else None
        pde_loss_gradient = (current_pde_loss - prev_pde_loss) if prev_pde_loss is not None else None
        prev_total_loss = current_total_loss
        prev_mass_loss = current_mass_loss
        prev_pde_loss = current_pde_loss

        print(
            f"[res={resolution}] iter={it:05d} "
            f"mass_loss={current_mass_loss:.8e} "
            f"pde_loss={current_pde_loss:.8e} "
            f"total_loss={current_total_loss:.8e} "
            f"delta_loss={(loss_delta if loss_delta is not None else float('nan')):.8e} "
            f"loss_gradient={(loss_gradient if loss_gradient is not None else float('nan')):.8e} "
            f"mass_loss_gradient={(mass_loss_gradient if mass_loss_gradient is not None else float('nan')):.8e} "
            f"pde_loss_gradient={(pde_loss_gradient if pde_loss_gradient is not None else float('nan')):.8e}"
        )

        if (
            converged_iter < 0
            and mass_loss_gradient is not None
            and pde_loss_gradient is not None
            and current_total_loss < config["converge_loss_threshold"]
            and abs(mass_loss_gradient) < config["converge_delta_threshold"]
            and abs(pde_loss_gradient) < config["converge_delta_threshold"]
        ):
            converged_iter = it
            print(
                f"[res={resolution}] converged at iter={it:05d}: "
                f"total_loss={current_total_loss:.8e} < loss_threshold={config['converge_loss_threshold']:.8e}, "
                f"|mass_grad|={abs(mass_loss_gradient):.8e}, "
                f"|pde_grad|={abs(pde_loss_gradient):.8e} < delta_threshold={config['converge_delta_threshold']:.8e}"
            )
            if config["stop_when_converged"]:
                break

    total_time = time.perf_counter() - start_train
    convergence_row = {
        "resolution": resolution,
        "converged": converged_iter > 0,
        "converged_iter": converged_iter if converged_iter > 0 else config["max_iters"],
        "train_time_s": total_time,
        "converge_loss_threshold": config["converge_loss_threshold"],
        "converge_delta_threshold": config["converge_delta_threshold"],
    }

    train_log_path = output_dir / f"physical_pretrain_log_res_{str(resolution).replace('.', 'p')}.csv"
    pd.DataFrame(train_rows).to_csv(train_log_path, index=False)
    return profile_row, convergence_row


def main():
    config = {
        "resolution": RESOLUTION,
        "batch_size": BATCH_SIZE,
        "max_iters": MAX_ITERS,
        "converge_loss_threshold": CONVERGE_LOSS_THRESHOLD,
        "converge_delta_threshold": CONVERGE_DELTA_THRESHOLD,
        "stop_when_converged": STOP_WHEN_CONVERGED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    }

    setup_seed(config["seed"])
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = config["resolution"]
    profile_row, convergence_row = physical_pretrain_for_resolution(resolution, config, device, output_dir)

    pd.DataFrame([profile_row]).to_csv(output_dir / "physical_step_profile.csv", index=False)
    pd.DataFrame([convergence_row]).to_csv(output_dir / "physical_convergence_summary.csv", index=False)
    print("Done; results written.")


if __name__ == "__main__":
    main()

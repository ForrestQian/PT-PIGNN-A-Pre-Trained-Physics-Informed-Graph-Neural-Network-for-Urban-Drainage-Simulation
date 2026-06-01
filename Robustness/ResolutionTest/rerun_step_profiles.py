from pathlib import Path

import pandas as pd
import torch

import PIGCN_Main as pm

N_REPEATS = 50
OUTPUT_BASE = ''  # Set locally before running


def main() -> None:
    base = Path(OUTPUT_BASE)
    resolution_folders = [
        (5.0, "5m_Resolution"),
        (2.0, "2m_Resolution"),
        (1.0, "1m_Resolution"),
        (0.5, "5dm_Resolution"),
        (0.2, "2dm_Resolution"),
        (0.1, "1dm_Resolution"),
        (0.05, "5cm_Resolution"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pm.setup_seed(pm.SEED)

    length_index, node_index, edge_index, adj_matrix, node_matrix, edge_matrix = pm.build_graph(device)
    num_edges = edge_index.shape[1]
    num_nodes = len(node_index)

    d_input = torch.tensor(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        dtype=torch.float32,
        device=device,
    )
    n_input = torch.tensor([0.009] * num_edges, dtype=torch.float32, device=device)
    s0_input = torch.tensor([0.005] * num_edges, dtype=torch.float32, device=device)
    criterion = torch.nn.L1Loss(reduction="mean").to(device)
    g = 9.8

    metric_cols = [
        "forward_time_s",
        "loss_compute_time_s",
        "backward_and_step_time_s",
        "mass_loss",
        "pde_loss",
        "total_loss",
        "forward_peak_memory_allocated_mb",
        "train_peak_memory_allocated_mb",
        "model_params_total",
        "model_params_trainable",
    ]

    for resolution, folder in resolution_folders:
        out_dir = base / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        run_rows = []
        refined_length_index = pm.build_refined_length_index(length_index, resolution)
        d_feature, n_feature, s0_feature = pm.build_physical_feature(
            pm.BATCH_SIZE, refined_length_index, d_input, n_input, s0_input, device
        )
        pde_label_1 = torch.zeros((pm.BATCH_SIZE, int(refined_length_index.sum().item()), 301), device=device)
        pde_label_2 = torch.zeros((pm.BATCH_SIZE, int(refined_length_index.sum().item()), 301), device=device)

        for k in range(N_REPEATS):
            print(f"Running resolution {resolution}m, repeat {k + 1}/{N_REPEATS}...")
            model = pm.build_model(
                resolution=resolution,
                length_index=length_index,
                node_index=node_index,
                edge_index=edge_index,
                adj_matrix=adj_matrix,
                node_matrix=node_matrix,
                edge_matrix=edge_matrix,
                device=device,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=pm.LR, weight_decay=pm.WEIGHT_DECAY)
            node_feature, edge_feature, q_feature = pm.random_physical_batch(pm.BATCH_SIZE, num_nodes, num_edges, d_input, device)

            pm.RESOLUTION = resolution
            row = pm.profile_one_step(
                model,
                optimizer,
                node_feature,
                edge_feature,
                q_feature,
                d_input,
                refined_length_index,
                d_feature,
                n_feature,
                s0_feature,
                pde_label_1,
                pde_label_2,
                criterion,
                g,
                device,
            )
            run_rows.append(row)

        runs_df = pd.DataFrame(run_rows)
        mean_row = runs_df[metric_cols].mean(numeric_only=True).to_dict()
        std_row = runs_df[metric_cols].std(numeric_only=True, ddof=1).to_dict()
        profile_row = {
            "resolution": resolution,
            "num_runs": N_REPEATS,
        }
        for col in metric_cols:
            profile_row[col] = float(mean_row.get(col, float("nan")))
            profile_row[f"{col}_std"] = float(std_row.get(col, float("nan")))

        target_path = out_dir / "physical_step_profile.csv"
        try:
            pd.DataFrame([profile_row]).to_csv(target_path, index=False)
            print(
                f"updated {folder}: "
                f"forward={profile_row['forward_time_s']:.6f}s±{profile_row['forward_time_s_std']:.6f}, "
                f"forward_peak={profile_row['forward_peak_memory_allocated_mb']:.1f}±{profile_row['forward_peak_memory_allocated_mb_std']:.1f}MB, "
                f"train_peak={profile_row['train_peak_memory_allocated_mb']:.1f}±{profile_row['train_peak_memory_allocated_mb_std']:.1f}MB"
            )
        except PermissionError:
            print(f"skipped {folder}: file is locked.")


if __name__ == "__main__":
    main()

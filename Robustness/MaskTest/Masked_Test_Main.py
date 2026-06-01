import os
import time
from itertools import combinations

import pandas as pd
import torch

from Function import nse_loss
from PIGCN_Train import (
    D_input,
    S0_input,
    _build_model,
    batch_size,
    dataloader,
    dev_edge_index,
    dev_node_index,
    dev_size,
    devloader,
    device,
    edge_index,
    epochs,
    length_index,
    lr,
    n_input,
    num_nodes,
    train_edge_index,
    train_node_index,
    weight_decay,
    PIGCN_train,
)


# =========================
# Edit here for PyCharm direct run
# =========================
MASK_NODES = []  # 1-based, e.g. [5, 8]
MASK_EDGES = []  # 1-based, e.g. [5, 14]
OUTPUT_ROOT = ''
RUN_SEED = 42
PHYSICAL_STEPS = 10000
DATA_MAX_STEPS_LOCAL = 10000
SEED_TIMEOUT_SEC = 3600
RUN_BATCH_PRESET = False  # True: batch baseline + pairwise node masks; False: single run with MASK_NODES/MASK_EDGES


def apply_inflow_flow_to_all_edges(edge_feature):
    """Replace all edge flow channels with inflow edge (edge 1) flow."""
    if edge_feature.shape[1] == 0:
        return edge_feature
    inflow_flow = edge_feature[:, 0:1, 0:1]
    edge_feature[:, :, 0:1] = inflow_flow
    return edge_feature


class EdgeInputAdapterLoader:
    """DataLoader adapter that rewrites edge_feature per rules."""

    def __init__(self, base_loader, use_inflow_for_all_edges=False):
        self.base_loader = base_loader
        self.use_inflow_for_all_edges = use_inflow_for_all_edges
        self.dataset = base_loader.dataset

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        for node_feature, edge_feature, output_node_feature, output_edge_feature in self.base_loader:
            if self.use_inflow_for_all_edges:
                edge_feature = edge_feature.clone()
                edge_feature = apply_inflow_flow_to_all_edges(edge_feature)
            yield node_feature, edge_feature, output_node_feature, output_edge_feature


def make_run_dir(output_root, mask_nodes, mask_edges):
    node_tag = "none" if not mask_nodes else "-".join(map(str, mask_nodes))
    edge_tag = "none" if not mask_edges else "-".join(map(str, mask_edges))
    base_name = f"mask_nodes_{node_tag}__edges_{edge_tag}"
    run_dir = os.path.join(output_root, base_name)
    idx = 2
    while os.path.exists(run_dir):
        run_dir = os.path.join(output_root, f"{base_name}_run{idx}")
        idx += 1
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def evaluate_with_mask(model_ckpt, mask_nodes, mask_edges, eval_loader, use_inflow_for_all_edges=False):
    model = _build_model().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    mask_node_zero = [i - 1 for i in mask_nodes]
    mask_edge_zero = [i - 1 for i in mask_edges]
    dev_node_zero = [i - 1 for i in dev_node_index]
    dev_edge_zero = [i - 1 for i in dev_edge_index]

    test_size = len(eval_loader.dataset)
    num_edges = edge_index.shape[1]

    all_pred_node = torch.zeros((test_size, num_nodes))
    all_true_node = torch.zeros((test_size, num_nodes))
    all_pred_edge = torch.zeros((test_size, num_edges, 2))
    all_true_edge = torch.zeros((test_size, num_edges, 2))

    ptr = 0
    with torch.no_grad():
        for data in eval_loader:
            node_feature, edge_feature, output_node_feature, output_edge_feature = data
            bsz = node_feature.shape[0]
            node_feature = node_feature.to(device)
            edge_feature = edge_feature.to(device)
            output_node_feature = output_node_feature.to(device)
            output_edge_feature = output_edge_feature.to(device)

            if use_inflow_for_all_edges:
                edge_feature = apply_inflow_flow_to_all_edges(edge_feature)

            for k in mask_node_zero:
                if 0 < k < node_feature.shape[1] - 1:
                    node_feature[:, k, :] = (node_feature[:, k - 1, :] + node_feature[:, k + 1, :]) / 2.0
            for k in mask_edge_zero:
                if 0 < k < edge_feature.shape[1] - 1:
                    edge_feature[:, k, :] = (edge_feature[:, k - 1, :] + edge_feature[:, k + 1, :]) / 2.0

            pred_node, pred_edge, _, _, _, _ = model(node_feature, edge_feature, D_input)
            all_pred_node[ptr:ptr + bsz, :] = pred_node.cpu()
            all_true_node[ptr:ptr + bsz, :] = output_node_feature.cpu()
            all_pred_edge[ptr:ptr + bsz, :, :] = pred_edge.cpu()
            all_true_edge[ptr:ptr + bsz, :, :] = output_edge_feature.cpu()
            ptr += bsz

    all_pred_node = all_pred_node[:, dev_node_zero]
    all_true_node = all_true_node[:, dev_node_zero]
    all_pred_edge = all_pred_edge[:, dev_edge_zero, :]
    all_true_edge = all_true_edge[:, dev_edge_zero, :]

    rows = []
    for i, node_id in enumerate(dev_node_index):
        rows.append(
            {
                "type": "node",
                "target_id": node_id,
                "metric": "NSE",
                "value": float(nse_loss(all_true_node[:, i], all_pred_node[:, i])),
            }
        )
    for i, edge_id in enumerate(dev_edge_index):
        rows.append(
            {
                "type": "edge_flow",
                "target_id": edge_id,
                "metric": "NSE",
                "value": float(nse_loss(all_true_edge[:, i, 0], all_pred_edge[:, i, 0])),
            }
        )
        rows.append(
            {
                "type": "edge_velocity",
                "target_id": edge_id,
                "metric": "NSE",
                "value": float(nse_loss(all_true_edge[:, i, 1], all_pred_edge[:, i, 1])),
            }
        )
    return pd.DataFrame(rows), all_pred_node, all_true_node, all_pred_edge, all_true_edge


def run_single_experiment(mask_nodes, mask_edges, tag="single"):
    run_dir = make_run_dir(OUTPUT_ROOT, mask_nodes, mask_edges)
    print(f"[{tag}] run started.", flush=True)

    train_nodes = [x for x in train_node_index if x not in mask_nodes]
    train_edges = [x for x in train_edge_index if x not in mask_edges]
    if not train_nodes:
        raise ValueError("No training nodes after masking; reduce mask-nodes")

    # If mask_edges empty or no train edges after masking, use inflow flow for all edges
    use_inflow_for_all_edges = (len(mask_edges) == 0) or (len(train_edges) == 0)
    if not train_edges:
        train_edges = list(train_edge_index)
        print(f"[{tag}] Note: no train edges after masking; enabled all-edge inflow flow mode and reset train_edge_index.", flush=True)
    train_loader_used = EdgeInputAdapterLoader(
        dataloader, use_inflow_for_all_edges=use_inflow_for_all_edges
    )
    dev_loader_used = EdgeInputAdapterLoader(
        devloader, use_inflow_for_all_edges=use_inflow_for_all_edges
    )

    cfg = {
        "mask_nodes": mask_nodes,
        "mask_edges": mask_edges,
        "use_inflow_for_all_edges": use_inflow_for_all_edges,
        "seed": RUN_SEED,
        "train_nodes_after_mask": train_nodes,
        "train_edges_after_mask": train_edges,
        "tag": tag,
    }
    pd.DataFrame([cfg]).to_csv(os.path.join(run_dir, "run_config.csv"), index=False)

    seed_deadline = time.time() + SEED_TIMEOUT_SEC
    phys_ckpt = os.path.join(run_dir, "PIGCN_phys_converged.pth")
    print(f"[{tag}] phase1: physical pre-training", flush=True)
    PIGCN_train(
        model=_build_model(),
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        dev_size=dev_size,
        D_input=D_input,
        n_input=n_input,
        S0_input=S0_input,
        epochs=epochs,
        physical_pre=True,
        physical_epochs=PHYSICAL_STEPS,
        train_node_index=train_nodes,
        train_edge_index=train_edges,
        dev_node_index=dev_node_index,
        dev_edge_index=dev_edge_index,
        length_index=length_index,
        device=device,
        edge_index=edge_index,
        dataloader=train_loader_used,
        devloader=dev_loader_used,
        num_nodes=num_nodes,
        resolution=1,
        pretrained_path="",
        physical_save_path=phys_ckpt,
        data_pretrained_path=None,
        run_seed=RUN_SEED,
        output_dir=run_dir,
        deadline_ts=seed_deadline,
    )

    print(f"[{tag}] phase2: data fine-tuning", flush=True)
    PIGCN_train(
        model=_build_model(),
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        dev_size=dev_size,
        D_input=D_input,
        n_input=n_input,
        S0_input=S0_input,
        epochs=epochs,
        physical_pre=False,
        physical_epochs=PHYSICAL_STEPS,
        data_max_steps=DATA_MAX_STEPS_LOCAL,
        train_node_index=train_nodes,
        train_edge_index=train_edges,
        dev_node_index=dev_node_index,
        dev_edge_index=dev_edge_index,
        length_index=length_index,
        device=device,
        edge_index=edge_index,
        dataloader=train_loader_used,
        devloader=dev_loader_used,
        num_nodes=num_nodes,
        resolution=1,
        pretrained_path="",
        physical_save_path=phys_ckpt,
        data_pretrained_path=phys_ckpt,
        run_seed=RUN_SEED,
        output_dir=run_dir,
        deadline_ts=seed_deadline,
    )

    data_ckpt = os.path.join(run_dir, "PIGCN_data_converged.pth")
    metrics_df, all_pred_node, all_true_node, all_pred_edge, all_true_edge = evaluate_with_mask(
        data_ckpt,
        mask_nodes=mask_nodes,
        mask_edges=mask_edges,
        eval_loader=dev_loader_used,
        use_inflow_for_all_edges=use_inflow_for_all_edges,
    )
    metrics_df.to_csv(os.path.join(run_dir, "masked_test_metrics.csv"), index=False)
    # Save full test predictions and ground truth
    pred_node_df = pd.DataFrame(all_pred_node.numpy())
    true_node_df = pd.DataFrame(all_true_node.numpy())
    pred_node_df.to_csv(os.path.join(run_dir, "test_pred_node_full.csv"), index=False)
    true_node_df.to_csv(os.path.join(run_dir, "test_true_node_full.csv"), index=False)

    num_eval_edges = all_pred_edge.shape[1]
    pred_edge_flow_df = pd.DataFrame(all_pred_edge[:, :, 0].numpy(), columns=[f"edge_{i+1}" for i in range(num_eval_edges)])
    true_edge_flow_df = pd.DataFrame(all_true_edge[:, :, 0].numpy(), columns=[f"edge_{i+1}" for i in range(num_eval_edges)])
    pred_edge_vel_df = pd.DataFrame(all_pred_edge[:, :, 1].numpy(), columns=[f"edge_{i+1}" for i in range(num_eval_edges)])
    true_edge_vel_df = pd.DataFrame(all_true_edge[:, :, 1].numpy(), columns=[f"edge_{i+1}" for i in range(num_eval_edges)])
    pred_edge_flow_df.to_csv(os.path.join(run_dir, "test_pred_edge_flow_full.csv"), index=False)
    true_edge_flow_df.to_csv(os.path.join(run_dir, "test_true_edge_flow_full.csv"), index=False)
    pred_edge_vel_df.to_csv(os.path.join(run_dir, "test_pred_edge_velocity_full.csv"), index=False)
    true_edge_vel_df.to_csv(os.path.join(run_dir, "test_true_edge_velocity_full.csv"), index=False)
    print(f"[{tag}] Masked test done; metrics saved.", flush=True)

    summary_row = {"tag": tag, "run_dir": run_dir, "mask_nodes": str(mask_nodes), "mask_edges": str(mask_edges)}
    for _, r in metrics_df.iterrows():
        key = f"{r['type']}_{int(r['target_id'])}_{r['metric']}"
        summary_row[key] = float(r["value"])
    pd.DataFrame([summary_row]).to_csv(os.path.join(run_dir, "experiment_summary.csv"), index=False)
    return summary_row


def build_batch_experiments():
    experiments = []
    # baseline: mask node 5,14 + edge 5
    experiments.append({"tag": "baseline_node_5_14__edge_5", "mask_nodes": [5, 14], "mask_edges": [5]})
    # Fixed mask edges 5,14 + single monitor masking
    all_nodes = [2, 5, 8, 11, 14, 17]
    for n in all_nodes:
        experiments.append(
            {
                "tag": f"single_node_{n}__edges_5_14",
                "mask_nodes": [n],
                "mask_edges": [5, 14],
            }
        )
    # Fixed mask edges 5,14 + pairwise monitor masking (order-free)
    for n1, n2 in combinations(all_nodes, 2):
        experiments.append(
            {
                "tag": f"pair_nodes_{n1}_{n2}__edges_5_14",
                "mask_nodes": [n1, n2],
                "mask_edges": [5, 14],
            }
        )
    return experiments


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if RUN_BATCH_PRESET:
        exp_list = build_batch_experiments()
        all_rows = []
        print(f"Batch experiments start, count: {len(exp_list)}", flush=True)
        for i, exp in enumerate(exp_list, 1):
            print("=" * 80, flush=True)
            print(f"[{i}/{len(exp_list)}] {exp['tag']} | nodes={exp['mask_nodes']} edges={exp['mask_edges']}", flush=True)
            row = run_single_experiment(
                mask_nodes=exp["mask_nodes"],
                mask_edges=exp["mask_edges"],
                tag=exp["tag"],
            )
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_ROOT, "batch_all_experiments_summary.csv"), index=False)
        print("Batch experiments done; summary saved: batch_all_experiments_summary.csv", flush=True)
    else:
        run_single_experiment(mask_nodes=MASK_NODES, mask_edges=MASK_EDGES, tag="single")


if __name__ == "__main__":
    main()

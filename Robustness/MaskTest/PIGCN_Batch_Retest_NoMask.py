import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from Function import nse_loss
from PIGCN_Train import (
    D_input,
    _build_model,
    dev_edge_index,
    dev_node_index,
    devloader,
    device,
    edge_index,
    num_nodes,
)


OUTPUT_ROOT = Path('')
CKPT_CANDIDATES = ("PIGCN_data_converged.pth", "PIGCN_phys_converged.pth")

# =========================
# Edit here for PyCharm direct run
# Empty string: process all subdirectories
# e.g.: "mask_nodes_2-17__edges_5-14"
# =========================
PYCHARM_SUBDIR = "mask_nodes_none__edges_none"


METRIC_COLUMNS = [
    "Node 2",
    "Node 5",
    "Node 8",
    "Node 11",
    "Node 14",
    "Node 17",
    "Link 5-6 Flow",
    "Link 5-6 Velocity",
    "Link 14-15 Flow",
    "Link 14-15 Velocity",
]


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location=device)


def pick_checkpoint(run_dir: Path) -> Path | None:
    for name in CKPT_CANDIDATES:
        p = run_dir / name
        if p.exists():
            return p
    return None


def evaluate_no_mask(model_ckpt: Path):
    model = _build_model().to(device)
    model.load_state_dict(load_checkpoint(model_ckpt))
    model.eval()

    dev_node_zero = [i - 1 for i in dev_node_index]
    dev_edge_zero = [i - 1 for i in dev_edge_index]
    test_size = len(devloader.dataset)
    num_edges = edge_index.shape[1]

    all_pred_node = torch.zeros((test_size, num_nodes))
    all_true_node = torch.zeros((test_size, num_nodes))
    all_pred_edge = torch.zeros((test_size, num_edges, 2))
    all_true_edge = torch.zeros((test_size, num_edges, 2))

    ptr = 0
    with torch.no_grad():
        for node_feature, edge_feature, output_node_feature, output_edge_feature in devloader:
            bsz = node_feature.shape[0]
            node_feature = node_feature.to(device)
            edge_feature = edge_feature.to(device)
            output_node_feature = output_node_feature.to(device)
            output_edge_feature = output_edge_feature.to(device)

            pred_node, pred_edge, _, _, _, _ = model(node_feature, edge_feature, D_input)
            all_pred_node[ptr:ptr + bsz, :] = pred_node.cpu()
            all_true_node[ptr:ptr + bsz, :] = output_node_feature.cpu()
            all_pred_edge[ptr:ptr + bsz, :, :] = pred_edge.cpu()
            all_true_edge[ptr:ptr + bsz, :, :] = output_edge_feature.cpu()
            ptr += bsz

    pred_node_dev = all_pred_node[:, dev_node_zero]
    true_node_dev = all_true_node[:, dev_node_zero]
    pred_edge_dev = all_pred_edge[:, dev_edge_zero, :]
    true_edge_dev = all_true_edge[:, dev_edge_zero, :]

    nse_node = nse_loss(true_node_dev, pred_node_dev)
    mae_node = torch.mean(torch.abs(true_node_dev - pred_node_dev), dim=0)
    rmse_node = torch.sqrt(torch.mean((true_node_dev - pred_node_dev) ** 2, dim=0))
    mape_node = torch.mean(torch.abs(true_node_dev - pred_node_dev) / torch.clamp(torch.abs(true_node_dev), min=1e-8), dim=0)

    nse_edge_flow = nse_loss(true_edge_dev[:, :, 0], pred_edge_dev[:, :, 0])
    mae_edge_flow = torch.mean(torch.abs(true_edge_dev[:, :, 0] - pred_edge_dev[:, :, 0]), dim=0)
    rmse_edge_flow = torch.sqrt(torch.mean((true_edge_dev[:, :, 0] - pred_edge_dev[:, :, 0]) ** 2, dim=0))
    mape_edge_flow = torch.mean(
        torch.abs(true_edge_dev[:, :, 0] - pred_edge_dev[:, :, 0]) / torch.clamp(torch.abs(true_edge_dev[:, :, 0]), min=1e-8),
        dim=0,
    )

    nse_edge_vel = nse_loss(true_edge_dev[:, :, 1], pred_edge_dev[:, :, 1])
    mae_edge_vel = torch.mean(torch.abs(true_edge_dev[:, :, 1] - pred_edge_dev[:, :, 1]), dim=0)
    rmse_edge_vel = torch.sqrt(torch.mean((true_edge_dev[:, :, 1] - pred_edge_dev[:, :, 1]) ** 2, dim=0))
    mape_edge_vel = torch.mean(
        torch.abs(true_edge_dev[:, :, 1] - pred_edge_dev[:, :, 1]) / torch.clamp(torch.abs(true_edge_dev[:, :, 1]), min=1e-8),
        dim=0,
    )

    row_nse = [
        nse_node[0].item(),
        nse_node[1].item(),
        nse_node[2].item(),
        nse_node[3].item(),
        nse_node[4].item(),
        nse_node[5].item(),
        nse_edge_flow[0].item(),
        nse_edge_vel[0].item(),
        nse_edge_flow[1].item(),
        nse_edge_vel[1].item(),
    ]
    row_mape = [
        mape_node[0].item(),
        mape_node[1].item(),
        mape_node[2].item(),
        mape_node[3].item(),
        mape_node[4].item(),
        mape_node[5].item(),
        mape_edge_flow[0].item(),
        mape_edge_vel[0].item(),
        mape_edge_flow[1].item(),
        mape_edge_vel[1].item(),
    ]
    row_mae = [
        mae_node[0].item(),
        mae_node[1].item(),
        mae_node[2].item(),
        mae_node[3].item(),
        mae_node[4].item(),
        mae_node[5].item(),
        mae_edge_flow[0].item(),
        mae_edge_vel[0].item(),
        mae_edge_flow[1].item(),
        mae_edge_vel[1].item(),
    ]
    row_rmse = [
        rmse_node[0].item(),
        rmse_node[1].item(),
        rmse_node[2].item(),
        rmse_node[3].item(),
        rmse_node[4].item(),
        rmse_node[5].item(),
        rmse_edge_flow[0].item(),
        rmse_edge_vel[0].item(),
        rmse_edge_flow[1].item(),
        rmse_edge_vel[1].item(),
    ]

    return (row_nse, row_mape, row_mae, row_rmse, all_pred_node, all_true_node, all_pred_edge, all_true_edge)


def overwrite_results(run_dir: Path, eval_out) -> None:
    row_nse, row_mape, row_mae, row_rmse, all_pred_node, all_true_node, all_pred_edge, all_true_edge = eval_out

    pd.DataFrame([row_nse], columns=METRIC_COLUMNS).to_csv(run_dir / "nse_all_Data.csv")
    pd.DataFrame([row_mape], columns=METRIC_COLUMNS).to_csv(run_dir / "mape_all_Data.csv")
    pd.DataFrame([row_mae], columns=METRIC_COLUMNS).to_csv(run_dir / "mae_all_Data.csv")
    pd.DataFrame([row_rmse], columns=METRIC_COLUMNS).to_csv(run_dir / "rmse_all_Data.csv")

    pred_node_df = pd.DataFrame(all_pred_node.numpy())
    true_node_df = pd.DataFrame(all_true_node.numpy())
    pred_node_df.to_csv(run_dir / "test_pred_node_full.csv", index=False)
    true_node_df.to_csv(run_dir / "test_true_node_full.csv", index=False)

    num_eval_edges = all_pred_edge.shape[1]
    pred_edge_flow_df = pd.DataFrame(all_pred_edge[:, :, 0].numpy(), columns=[f"edge_{i + 1}" for i in range(num_eval_edges)])
    true_edge_flow_df = pd.DataFrame(all_true_edge[:, :, 0].numpy(), columns=[f"edge_{i + 1}" for i in range(num_eval_edges)])
    pred_edge_vel_df = pd.DataFrame(all_pred_edge[:, :, 1].numpy(), columns=[f"edge_{i + 1}" for i in range(num_eval_edges)])
    true_edge_vel_df = pd.DataFrame(all_true_edge[:, :, 1].numpy(), columns=[f"edge_{i + 1}" for i in range(num_eval_edges)])
    pred_edge_flow_df.to_csv(run_dir / "test_pred_edge_flow_full.csv", index=False)
    true_edge_flow_df.to_csv(run_dir / "test_true_edge_flow_full.csv", index=False)
    pred_edge_vel_df.to_csv(run_dir / "test_pred_edge_velocity_full.csv", index=False)
    true_edge_vel_df.to_csv(run_dir / "test_true_edge_velocity_full.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch (or single dir): reload converged models, retest without mask, overwrite results.")
    parser.add_argument(
        "--subdir",
        type=str,
        default="",
        help="Process only this subfolder, e.g. mask_nodes_2-17__edges_5-14; empty = all subdirectories.",
    )
    args = parser.parse_args()

    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError("Output directory not found.")

    selected_subdir = args.subdir.strip() if args.subdir else PYCHARM_SUBDIR.strip()

    if selected_subdir:
        target = OUTPUT_ROOT / selected_subdir
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError("Specified subdirectory not found.")
        run_dirs = [target]
        print(f"Processing subdirectory only: {target.name}")
    else:
        run_dirs = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir()]
        run_dirs.sort(key=lambda x: x.name)
        print(f"Batch processing subdirectory count: {len(run_dirs)}")

    ok = 0
    skipped = 0
    failed = 0

    for run_dir in run_dirs:
        ckpt = pick_checkpoint(run_dir)
        if ckpt is None:
            print(f"[SKIP] {run_dir.name} missing converged model file")
            skipped += 1
            continue
        try:
            eval_out = evaluate_no_mask(ckpt)
            overwrite_results(run_dir, eval_out)
            print(f"[OK] {run_dir.name} overwrote test results.")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {run_dir.name} processing failed: {e}")
            failed += 1

    print("-" * 72)
    print(f"Done: success {ok} | skip {skipped} | failed {failed}")


if __name__ == "__main__":
    main()

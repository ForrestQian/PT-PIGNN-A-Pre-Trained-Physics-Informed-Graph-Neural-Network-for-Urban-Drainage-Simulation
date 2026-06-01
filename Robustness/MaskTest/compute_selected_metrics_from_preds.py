import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# =========================
# Edit here for PyCharm direct run
# =========================
MASK_EXPERIMENTS_ROOT = Path('')
TARGET_SUBDIR = "mask_nodes_14__edges_5-14"
# Two formats supported:
# 1) By column index (0-based): "1,4,10"
# 2) By column name: "Node 2,Node 11" or "edge_5,edge_14"
NODE_SELECT = "1,4,7,10,13,16"
EDGE_SELECT = "4,13"
OUTPUT_FILE_NAME = "selected_metrics_result.csv"
OVERWRITE_METRIC_CSVS = True
PRINT_ALL_NODE_AND_FLOW_NSE = True
# Custom name map (key=original column, value=display name)
# Default node columns are "0"..."18"; default edge columns are "edge_1"..."edge_18"
NODE_NAME_MAP = {
    "0": "Node 1",
    "1": "Node 2",
    "2": "Node 3",
    "3": "Node 4",
    "4": "Node 5",
    "5": "Node 6",
    "6": "Node 7",
    "7": "Node 8",
    "8": "Node 9",
    "9": "Node 10",
    "10": "Node 11",
    "11": "Node 12",
    "12": "Node 13",
    "13": "Node 14",
    "14": "Node 15",
    "15": "Node 16",
    "16": "Node 17",
    "17": "Node 18",
    "18": "Node 19",
}
EDGE_NAME_MAP = {
    "edge_1": "Link 1-2",
    "edge_2": "Link 2-3",
    "edge_3": "Link 3-4",
    "edge_4": "Link 4-5",
    "edge_5": "Link 5-6",
    "edge_6": "Link 6-7",
    "edge_7": "Link 7-8",
    "edge_8": "Link 8-9",
    "edge_9": "Link 9-10",
    "edge_10": "Link 10-11",
    "edge_11": "Link 11-12",
    "edge_12": "Link 12-13",
    "edge_13": "Link 13-14",
    "edge_14": "Link 14-15",
    "edge_15": "Link 15-16",
    "edge_16": "Link 16-17",
    "edge_17": "Link 17-18",
    "edge_18": "Link 18-19",
}


def nse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.mean((y_true - np.mean(y_true)) ** 2)
    if var == 0:
        return np.nan
    return float(1.0 - mse / var)


def parse_select_tokens(select_text: str) -> List[str]:
    if not select_text.strip():
        return []
    return [s.strip() for s in select_text.split(",") if s.strip()]


def resolve_columns(df: pd.DataFrame, tokens: List[str], kind: str) -> List[str]:
    if not tokens:
        return []
    cols = list(df.columns)
    resolved: List[str] = []
    for t in tokens:
        if t.isdigit():
            idx = int(t)
            if idx < 0 or idx >= len(cols):
                raise ValueError(f"{kind} selection index out of range: {t}, valid range 0~{len(cols)-1}")
            resolved.append(cols[idx])
        else:
            if t not in cols:
                raise ValueError(f"{kind} column name not found: {t}")
            resolved.append(t)
    return resolved


def calc_metrics_per_column(true_df: pd.DataFrame, pred_df: pd.DataFrame, columns: List[str], group: str) -> pd.DataFrame:
    rows = []
    for c in columns:
        y_true = true_df[c].to_numpy(dtype=float)
        y_pred = pred_df[c].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        denom = np.clip(np.abs(y_true), 1e-12, None)
        mape = float(np.mean(np.abs(y_true - y_pred) / denom))
        nse = nse_np(y_true, y_pred)
        rows.append(
            {
                "group": group,
                "column": c,
                "nse": nse,
                "mae": mae,
                "mape": mape,
                "rmse": rmse,
            }
        )
    return pd.DataFrame(rows)


def build_name_map(cols: List[str], user_map: dict) -> dict:
    out = {}
    for c in cols:
        out[c] = user_map.get(c, c)
    return out


def build_metric_csv_rows(result_df: pd.DataFrame) -> dict:
    nse_row = {}
    mae_row = {}
    mape_row = {}
    rmse_row = {}

    for _, r in result_df.iterrows():
        if r["group"] == "node":
            col_name = r["display_name"]
        elif r["group"] == "edge_flow":
            col_name = f"{r['display_name']} Flow"
        elif r["group"] == "edge_velocity":
            col_name = f"{r['display_name']} Velocity"
        else:
            col_name = str(r["display_name"])

        nse_row[col_name] = float(r["nse"])
        mae_row[col_name] = float(r["mae"])
        mape_row[col_name] = float(r["mape"])
        rmse_row[col_name] = float(r["rmse"])

    return {
        "nse_all_Data.csv": pd.DataFrame([nse_row]),
        "mae_all_Data.csv": pd.DataFrame([mae_row]),
        "mape_all_Data.csv": pd.DataFrame([mape_row]),
        "rmse_all_Data.csv": pd.DataFrame([rmse_row]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NSE/MAE/MAPE/RMSE from pred/target files in subdirectory by selected columns.")
    parser.add_argument("--subdir", type=str, default="", help="Subdirectory name, e.g. mask_nodes_2-17__edges_5-14")
    parser.add_argument("--node-select", type=str, default="", help="Node columns: indices or names, comma-separated")
    parser.add_argument("--edge-select", type=str, default="", help="Edge columns: indices or names, comma-separated")
    parser.add_argument("--output", type=str, default="", help="Output filename (written under subdirectory)")
    args = parser.parse_args()

    subdir = args.subdir.strip() if args.subdir.strip() else TARGET_SUBDIR
    node_select = args.node_select.strip() if args.node_select.strip() else NODE_SELECT
    edge_select = args.edge_select.strip() if args.edge_select.strip() else EDGE_SELECT
    output_file_name = args.output.strip() if args.output.strip() else OUTPUT_FILE_NAME

    run_dir = MASK_EXPERIMENTS_ROOT / subdir
    if not run_dir.exists():
        raise FileNotFoundError("Subdirectory not found.")

    pred_node = pd.read_csv(run_dir / "test_pred_node_full.csv")
    true_node = pd.read_csv(run_dir / "test_true_node_full.csv")
    pred_edge_flow = pd.read_csv(run_dir / "test_pred_edge_flow_full.csv")
    true_edge_flow = pd.read_csv(run_dir / "test_true_edge_flow_full.csv")
    pred_edge_vel = pd.read_csv(run_dir / "test_pred_edge_velocity_full.csv")
    true_edge_vel = pd.read_csv(run_dir / "test_true_edge_velocity_full.csv")

    if PRINT_ALL_NODE_AND_FLOW_NSE:
        all_node_rows = calc_metrics_per_column(true_node, pred_node, list(pred_node.columns), group="node")
        all_flow_rows = calc_metrics_per_column(true_edge_flow, pred_edge_flow, list(pred_edge_flow.columns), group="edge_flow")
        node_name_map_all = build_name_map(list(pred_node.columns), NODE_NAME_MAP)
        edge_name_map_all = build_name_map(list(pred_edge_flow.columns), EDGE_NAME_MAP)

        print("NSE for all nodes:")
        for _, r in all_node_rows.iterrows():
            display = node_name_map_all.get(r["column"], r["column"])
            print(f"  - {display}: {r['nse']:.6f}")

        print("NSE for all pipe flows:")
        for _, r in all_flow_rows.iterrows():
            display = edge_name_map_all.get(r["column"], r["column"])
            print(f"  - {display} Flow: {r['nse']:.6f}")

    node_tokens = parse_select_tokens(node_select)
    edge_tokens = parse_select_tokens(edge_select)
    if not node_tokens and not edge_tokens:
        raise ValueError("Select at least node or edge columns")

    node_cols = resolve_columns(pred_node, node_tokens, kind="node")
    edge_cols = resolve_columns(pred_edge_flow, edge_tokens, kind="edge")

    pieces = []
    if node_cols:
        pieces.append(calc_metrics_per_column(true_node, pred_node, node_cols, group="node"))
    if edge_cols:
        pieces.append(calc_metrics_per_column(true_edge_flow, pred_edge_flow, edge_cols, group="edge_flow"))
        pieces.append(calc_metrics_per_column(true_edge_vel, pred_edge_vel, edge_cols, group="edge_velocity"))

    result = pd.concat(pieces, ignore_index=True)

    node_name_map = build_name_map(node_cols, NODE_NAME_MAP)
    edge_name_map = build_name_map(edge_cols, EDGE_NAME_MAP)
    result["display_name"] = result.apply(
        lambda r: (
            node_name_map.get(r["column"], r["column"])
            if r["group"] == "node"
            else edge_name_map.get(r["column"], r["column"])
        ),
        axis=1,
    )

    out_path = run_dir / output_file_name
    result.to_csv(out_path, index=False)
    print("Metrics written.")

    if OVERWRITE_METRIC_CSVS:
        metric_csv_map = build_metric_csv_rows(result)
        for filename, df_metric in metric_csv_map.items():
            target_path = run_dir / filename
            df_metric.to_csv(target_path)
            print("Target metrics overwritten.")

    print("Computed columns:")
    for _, row in result[["group", "column", "display_name"]].drop_duplicates().iterrows():
        print(f"  - {row['group']}: {row['column']} ({row['display_name']})")


if __name__ == "__main__":
    main()

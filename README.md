# PT-PIGNN-CODE

Code repository for **Physics-informed Transfer Learning Graph Neural Network (PT-PIGNN)**.

## Scope and Data Policy

This repository publishes the **core research code** only:

- Model **training**
- Model **testing / evaluation**
- **Hyperparameter tuning** (Hyperopt scripts in `Kunshan/Hyperparameter/`)

It does **not** include the original Kunshan/Changzhou sewer-network datasets, SWMM exports, trained checkpoints, or experiment artifacts. Those materials contain **operationally sensitive information** and cannot be released publicly. All hard-coded local load paths have been removed from the scripts; you must supply your own private data and set paths locally before running.

---

## Repository Layout

```
PT-PIGNN-CODE/
├── Kunshan/                    # Case study 1: Kunshan drainage network
│   ├── GNN/                    # Data-driven GNN baseline
│   ├── PIGNN/                  # Physics-informed GNN (Saint-Venant equations)
│   ├── PT-PIGNN/               # Pretrain + fine-tune PT-PIGNN
│   ├── PINN/                   # Physics-informed NN baseline
│   └── Hyperparameter/         # Hyperparameter search (Hyperopt)
│
├── Changzhou/                  # Case study 2: Changzhou drainage network (complex system)
│   ├── GNN/                    # Data-driven GNN baseline
│   ├── PIGNN-Mass/             # PIGNN with mass conservation only
│   ├── PIGNN-SVE/              # PIGNN with full Saint-Venant equations
│   ├── PT-PIGNN-Mass/          # PT-PIGNN with mass-conservation pretraining
│   └── PT-PIGNN-SVE/           # PT-PIGNN with SVE pretraining
│
└── Robustness/                 # Robustness and sensitivity analysis
    ├── MaskTest/               # Missing-observation (masking) experiments
    ├── SeedTest/               # Random-seed stability experiments
    └── ResolutionTest/         # Spatial-resolution sensitivity experiments
```

---

## Configuring Local Paths

Because raw network data are not shipped with this repo, configure paths in the relevant entry script before execution:

| Case | Where to configure | What to provide |
|------|-------------------|-----------------|
| Kunshan | `file_path` in `GCN_Dataset` / `GNN_Dataset` / `PINN_Dataset`, plus `torch.load(...)` in train/test scripts | Private Excel workbooks with sheets `Node_Depth`, `Node_Inflow`, `Link_Flow`, `Link_Velocity` |
| Changzhou | Arguments to `dataset_create(...)` in each `Main_*.py` | Private graph archive and CSV time series; set `sensor_nodes`, `sensor_links`, `mask_nodes`, `mask_links` in `dataset_create.py` |
| Robustness | `file_path` in `GCN_Dataset`, `OUTPUT_ROOT` / `RESULT_ROOT`, optional `pretrained_path` | Same Kunshan-style private Excel layout plus local output directories |
| Hyperparameter | `TRIALS_PATH`, `RESULTS_OUTPUT_PATH` in `Hyperparameter/` scripts | Locally saved Hyperopt trial files |

Empty path strings in the repository are intentional placeholders.

---

## Kunshan (`Kunshan/`)

Small benchmark network (19 nodes, 18 pipes). Data are read through dataset classes in `*_Net.py`.

### `GNN/`

| File | Role |
|------|------|
| `GNN_Net.py` | GNN architecture and `GNN_Dataset` |
| `Function.py` | Metrics, PDE helpers, weight init, subset utilities |
| `GNN_Train.py` | Standard GNN training loop |
| `GNN_Train_best.py` | Training with fixed best hyperparameters |
| `GNN_Test.py` | One-step prediction and evaluation on the test set |

**Entry point:** `python GNN_Train.py` or `python GNN_Train_best.py`, then `python GNN_Test.py`.

### `PIGNN/`

| File | Role |
|------|------|
| `PIGCN_Net.py` | PIGNN (physics-informed GCN) model |
| `Function.py` | Shared utilities (same role as in `GNN/`) |
| `PIGCN_Train.py` | End-to-end PIGNN training (data + PDE loss) |
| `PIGCN_Train_best.py` | Training with fixed best hyperparameters |
| `PIGCN_Test.py` | Test-set evaluation and NSE computation |

**Entry point:** `python PIGCN_Train.py` or `python PIGCN_Train_best.py`, then `python PIGCN_Test.py`.

### `PT-PIGNN/`

| File | Role |
|------|------|
| `PIGCN_Net.py` | PT-PIGNN model (same family as PIGNN) |
| `Function.py` | Shared utilities |
| `PIGCN_Train.py` | Two-stage training: physical pretraining (`physical_pre=True`) then data fine-tuning |
| `PIGCN_Train_best.py` | Same pipeline with fixed best hyperparameters |
| `PIGCN_Test.py` | Test-set evaluation |

**Entry point:** `python PIGCN_Train.py` (pretrain + fine-tune in one script), then `python PIGCN_Test.py`.

### `PINN/`

| File | Role |
|------|------|
| `PINN_Net.py` | Fully connected PINN model and dataset |
| `Function.py` | NSE, weight init, subset helpers |
| `PINN_Train.py` | PINN training with data and PDE losses |
| `PINN_Test.py` | PINN test-set evaluation |

**Entry point:** `python PINN_Train.py`, then `python PINN_Test.py`.

### `Hyperparameter/`

| File | Role |
|------|------|
| `Hyper_GNN.py` | Hyperopt search for GNN hyperparameters |
| `Hyper_PIGCN.py` | Hyperopt search for PIGNN hyperparameters |
| `Hyperout.py` | Load saved Trials and export top hyperparameter sets |

**Entry point:** set `TRIALS_PATH`, run `python Hyper_GNN.py` or `python Hyper_PIGCN.py`, then analyze with `python Hyperout.py`.

---

## Changzhou (`Changzhou/`)

Large SWMM-based network. Each subfolder is self-contained and shares `dataset_create.py`, `Function.py`, a model net file, and a `Main_*.py` entry script.

### Shared files (in every subfolder)

| File | Role |
|------|------|
| `dataset_create.py` | Convert private graph/CSV inputs to cached PyTorch DataLoaders |
| `Function.py` | Adjacency construction, normalization, PDE loss, NSE, utilities |
| `GNN_Net.py` or `PIGNN_Net.py` | Model definition for that experiment type |

### Subfolders

| Folder | Entry script | Description |
|--------|--------------|-------------|
| `GNN/` | `Main_GNN.py` | Data-driven GNN baseline |
| `PIGNN-Mass/` | `Main_PIGNN.py` | PIGNN with mass-conservation PDE loss |
| `PIGNN-SVE/` | `Main_PIGNN.py` | PIGNN with full Saint-Venant PDE loss |
| `PT-PIGNN-Mass/` | `Main_PT_PIGNN.py` | Mass pretrain → checkpoint load → fine-tune → test |
| `PT-PIGNN-SVE/` | `Main_PT_PIGNN.py` | SVE pretrain → optional fine-tune → test |

---

## Robustness (`Robustness/`)

Kunshan-scale network setup (18 pipes). Each test folder duplicates `PIGCN_Net.py`, `Function.py`, and train/test helpers so experiments stay isolated.

### `MaskTest/` — missing observations

| File | Role |
|------|------|
| `Masked_Test_Main.py` | Main driver: mask nodes/edges, pretrain, fine-tune, save metrics |
| `PIGCN_Train.py` | Reusable two-stage train function |
| `PIGCN_Test.py` | Standard test-set evaluation |
| `PIGCN_Net.py` | Model definition |
| `Function.py` | Shared utilities |
| `PIGCN_Batch_Retest_NoMask.py` | Reload converged checkpoints and retest without masking |
| `compute_selected_metrics_from_preds.py` | Compute NSE/MAE/MAPE/RMSE for selected columns from saved preds |

**Entry point:** configure `MASK_NODES` / `MASK_EDGES` and `OUTPUT_ROOT` in `Masked_Test_Main.py`, then `python Masked_Test_Main.py`.

### `SeedTest/` — random seed stability

| File | Role |
|------|------|
| `SeedTest_Main.py` | Run 100 seeds; physical pretrain + fine-tune; write per-seed and summary CSVs |
| `PIGCN_Test.py` | Test-set evaluation helper |
| `PIGCN_Net.py` | Model definition |
| `Function.py` | Shared utilities |

**Entry point:** set `RESULT_ROOT` and data paths, then `python SeedTest_Main.py`.

### `ResolutionTest/` — spatial resolution sensitivity

| File | Role |
|------|------|
| `Resolution_Main.py` | Train/evaluate at configurable pipe discretization resolution |
| `PIGCN_Train.py` | Resolution-aware training script |
| `PIGCN_Test.py` | Test-set evaluation |
| `PIGCN_Net.py` | Model with resolution-dependent edge CNN |
| `Function.py` | Shared utilities |
| `rerun_step_profiles.py` | Re-run step-profile analysis from saved resolution outputs |

**Entry point:** set `OUTPUT_DIR`, `RESOLUTION`, and data paths in `Resolution_Main.py`, then `python Resolution_Main.py`.

---

## Common File Naming Conventions

| Pattern | Purpose |
|---------|---------|
| `*_Net.py` | Model architecture and dataset classes |
| `*_Train.py` / `Main_*.py` | Training entry scripts |
| `*_Test.py` | Testing and metric computation |
| `*_Train_best.py` | Training with fixed optimal hyperparameters (Kunshan) |
| `Function.py` | Shared helpers: normalization, metrics, PDE terms |
| `dataset_create.py` | Graph/time-series → DataLoaders (Changzhou only) |
| `Hyper_*.py` | Hyperopt hyperparameter search (Kunshan) |

---

## Dependencies

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the third-party packages below. Some packages appear more than once with different version pins (notably `torch`, `networkx`, `sympy`, and `tqdm`); keep **one** compatible set for your OS/CUDA setup before installing—especially for PyTorch.

| Package | Pinned version(s) in `requirements.txt` | Used for |
|---------|----------------------------------------|----------|
| `torch` | 2.4.0+cu124, 2.6.0, 2.8.0, 2.9.1 | Model training and inference |
| `numpy` | 2.4.6 | Numerical arrays |
| `pandas` | 3.0.3 | Tabular data I/O and metrics logging |
| `matplotlib` | 3.8.4 | Plotting (where used) |
| `networkx` | 3.2.1 / 3.4.2 / 3.6.1 | Graph utilities |
| `sympy` | 1.12 / 1.13.1 / 1.14.0 | Symbolic math (where used) |
| `tqdm` | 4.66.4 / 4.67.1 / 4.67.3 | Progress bars |
| `hyperopt` | 0.2.7 | Hyperparameter search (`Kunshan/Hyperparameter/`) |
| `visdom` | 0.2.4 | Live training visualization (optional) |
| `h5py` | 3.11.0 | HDF5 I/O (optional / commented examples in test scripts) |

**Recommended environment:** Python 3.10+ (compatible with the pinned `pandas` / `numpy` stack). Install a **single** PyTorch build matching your hardware, for example from [pytorch.org](https://pytorch.org), then install the remaining packages from `requirements.txt`.

---

## Quick Start

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Obtain or simulate **private** sewer-network data compatible with the expected tensor/graph layout.
3. Set all required path placeholders (`file_path`, `dataset_create(...)` arguments, checkpoint paths, output roots) in the target script.
4. Run training, for example:
   ```bash
   cd Kunshan/PT-PIGNN
   python PIGCN_Train.py

   cd Changzhou/PT-PIGNN-Mass
   python Main_PT_PIGNN.py

   cd Robustness/SeedTest
   python SeedTest_Main.py
   ```
5. Run the corresponding `*_Test.py` or read metrics produced by `Main_*.py` / robustness drivers.
6. Optional: start Visdom (`visdom -port 8097`) when a script uses `Visdom(...)`.

---

## Citation

If you use this code, please cite the associated PT-PIGNN paper (DOI to be added).

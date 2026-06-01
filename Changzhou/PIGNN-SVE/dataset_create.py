import numpy as np
import pandas as pd
import torch
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from matplotlib import pyplot as plt
class TimeSeriesGraphDataset(Dataset):
    def __init__(self, level_df, inflow_df, flow_df, velocity_df, history_T, future_T, stride=1, cache=True):
        """
        Initialize time-series graph dataset

        Args:
        level_df: DataFrame of shape (total_time, N), node depth data
        inflow_df: DataFrame of shape (total_time, N), node inflow data
        flow_df: DataFrame of shape (total_time, M), edge flow data
        velocity_df: DataFrame of shape (total_time, M), edge velocity data
        history_T: history window length
        future_T: forecast horizon
        stride: sliding window stride
        cache: Cache all samples in memory (default True, speeds up training)
        """
        # Convert to numpy and transpose to (N, total_time) and (M, total_time)
        level_data = level_df.values.T  # (N, total_time)
        inflow_data = inflow_df.values.T  # (N, total_time)
        flow_data = flow_df.values.T  # (M, total_time)
        velocity_data = velocity_df.values.T  # (M, total_time)

        # Concat level and inflow -> node_input (N, total_time, 2)
        node_input = np.stack([level_data, inflow_data], axis=-1)  # (N, total_time, 2)
        # Concat flow and velocity -> edge_input (M, total_time, 2)
        edge_input = np.stack([flow_data, velocity_data], axis=-1)  # (M, total_time, 2)
        # target includes node level and edge flow/velocity
        node_target = level_data[..., np.newaxis]  # (N, total_time, 1)
        # edge flow and velocity as edge_target
        edge_target_1 = flow_data[..., np.newaxis]  # (M, total_time, 1)
        edge_target_2 = velocity_data[..., np.newaxis]  # (M, total_time, 1)
        edge_target = np.concatenate([edge_target_1, edge_target_2], axis=-1)  # (M, total_time, 2)

        self.history_T = history_T
        self.future_T = future_T
        self.stride = stride
        self.cache = cache

        # Compute total number of samples
        total_time = node_input.shape[1]
        self.num_samples = (total_time - history_T - future_T) // stride + 1

        # If cache enabled, precompute all samples
        if cache:
            print(f"Caching {self.num_samples} samples...")
            self.cached_samples = []
            for idx in range(self.num_samples):
                start_time = idx * stride
                # node_input: (N, history_T, 2)
                node_input_seq = node_input[:, start_time:start_time + history_T, :]
                # edge_input: (M, history_T, 2)
                edge_input_seq = edge_input[:, start_time:start_time + history_T, :]
                # target: node level and edge flow
                target_start = start_time + history_T
                node_target_seq = node_target[:, target_start:target_start + future_T, :]  # (N, future_T, 1)
                edge_target_seq = edge_target[:, target_start:target_start + future_T, :]  # (M, future_T, 1)
                
                # Convert to torch tensors and cache
                sample = {
                    'node_input': torch.FloatTensor(node_input_seq),  # (N, history_T, 2)
                    'edge_input': torch.FloatTensor(edge_input_seq),  # (M, history_T, 2)
                    'node_target': torch.FloatTensor(node_target_seq),  # (N, future_T, 1) - level
                    'edge_target': torch.FloatTensor(edge_target_seq)  # (M, future_T, 1) - flow
                }
                self.cached_samples.append(sample)
            print(f"Caching complete!")
        else:
            # Without cache, keep raw arrays
            self.node_input = node_input
            self.edge_input = edge_input
            self.node_target = node_target
            self.edge_target = edge_target

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cache:
            # Return directly from cache
            return self.cached_samples[idx]
        else:
            # Compute on the fly (no cache)
            start_time = idx * self.stride

            # node_input: (N, history_T, 2)
            node_input_seq = self.node_input[:, start_time:start_time + self.history_T, :]

            # edge_input: (M, history_T, 2)
            edge_input_seq = self.edge_input[:, start_time:start_time + self.history_T, :]

            # target: node level and edge flow
            target_start = start_time + self.history_T
            node_target_seq = self.node_target[:, target_start:target_start + self.future_T, :]  # (N, future_T, 1)
            edge_target_seq = self.edge_target[:, target_start:target_start + self.future_T, :]  # (M, future_T, 1)

            # Convert to torch tensor
            node_input_tensor = torch.FloatTensor(node_input_seq)
            edge_input_tensor = torch.FloatTensor(edge_input_seq)
            node_target_tensor = torch.FloatTensor(node_target_seq)
            edge_target_tensor = torch.FloatTensor(edge_target_seq)
            return {
                'node_input': node_input_tensor,  # (N, history_T, 2)
                'edge_input': edge_input_tensor,  # (M, history_T, 2)
                'node_target': node_target_tensor,  # (N, future_T, 1) - level
                'edge_target': edge_target_tensor  # (M, future_T, 1) - flow
            }


class TimeSeriesGraphDataset_Test(Dataset):
    def __init__(self, level_df, target_level_df, inflow_df, flow_df, target_flow_df, velocity_df, target_velocity_df,
                 history_T, future_T, stride=1, cache=True):
        """
        Initialize time-series graph dataset

        Args:
        level_df: DataFrame of shape (total_time, N), node depth data
        inflow_df: DataFrame of shape (total_time, N), node inflow data
        flow_df: DataFrame of shape (total_time, M), edge flow data
        velocity_df: DataFrame of shape (total_time, M), edge velocity data
        history_T: history window length
        future_T: forecast horizon
        stride: sliding window stride
        cache: Cache all samples in memory (default True, speeds up training)
        """
        # Convert to numpy and transpose to (N, total_time) and (M, total_time)
        level_data = level_df.values.T  # (N, total_time)
        target_level_data = target_level_df.values.T  # (N, total_time)
        inflow_data = inflow_df.values.T  # (N, total_time)
        flow_data = flow_df.values.T  # (M, total_time)
        target_flow_data = target_flow_df.values.T  # (M, total_time)
        velocity_data = velocity_df.values.T  # (M, total_time)
        target_velocity_data = target_velocity_df.values.T  # (M, total_time)

        # Concat level and inflow -> node_input (N, total_time, 2)
        node_input = np.stack([level_data, inflow_data], axis=-1)  # (N, total_time, 2)
        # Concat flow and velocity -> edge_input (M, total_time, 2)
        edge_input = np.stack([flow_data, velocity_data], axis=-1)  # (M, total_time, 2)
        # target includes node level and edge flow/velocity
        node_target = target_level_data[..., np.newaxis]  # (N, total_time, 1)
        # edge flow and velocity as edge_target
        edge_target_1 = target_flow_data[..., np.newaxis]  # (M, total_time, 1)
        edge_target_2 = target_velocity_data[..., np.newaxis]  # (M, total_time, 1)
        edge_target = np.concatenate([edge_target_1, edge_target_2], axis=-1)  # (M, total_time, 2)

        self.history_T = history_T
        self.future_T = future_T
        self.stride = stride
        self.cache = cache

        # Compute total number of samples
        total_time = node_input.shape[1]
        self.num_samples = (total_time - history_T - future_T) // stride + 1

        # If cache enabled, precompute all samples
        if cache:
            print(f"Caching {self.num_samples} samples...")
            self.cached_samples = []
            for idx in range(self.num_samples):
                start_time = idx * stride
                # node_input: (N, history_T, 2)
                node_input_seq = node_input[:, start_time:start_time + history_T, :]
                # edge_input: (M, history_T, 2)
                edge_input_seq = edge_input[:, start_time:start_time + history_T, :]
                # target: node level and edge flow
                target_start = start_time + history_T
                node_target_seq = node_target[:, target_start:target_start + future_T, :]  # (N, future_T, 1)
                edge_target_seq = edge_target[:, target_start:target_start + future_T, :]  # (M, future_T, 1)

                # Convert to torch tensors and cache
                sample = {
                    'node_input': torch.FloatTensor(node_input_seq),  # (N, history_T, 2)
                    'edge_input': torch.FloatTensor(edge_input_seq),  # (M, history_T, 2)
                    'node_target': torch.FloatTensor(node_target_seq),  # (N, future_T, 1) - level
                    'edge_target': torch.FloatTensor(edge_target_seq)  # (M, future_T, 1) - flow
                }
                self.cached_samples.append(sample)
            print(f"Caching complete!")
        else:
            # Without cache, keep raw arrays
            self.node_input = node_input
            self.edge_input = edge_input
            self.node_target = node_target
            self.edge_target = edge_target

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cache:
            # Return directly from cache
            return self.cached_samples[idx]
        else:
            # Compute on the fly (no cache)
            start_time = idx * self.stride

            # node_input: (N, history_T, 2)
            node_input_seq = self.node_input[:, start_time:start_time + self.history_T, :]

            # edge_input: (M, history_T, 2)
            edge_input_seq = self.edge_input[:, start_time:start_time + self.history_T, :]

            # target: node level and edge flow
            target_start = start_time + self.history_T
            node_target_seq = self.node_target[:, target_start:target_start + self.future_T, :]  # (N, future_T, 1)
            edge_target_seq = self.edge_target[:, target_start:target_start + self.future_T, :]  # (M, future_T, 1)

            # Convert to torch tensor
            node_input_tensor = torch.FloatTensor(node_input_seq)
            edge_input_tensor = torch.FloatTensor(edge_input_seq)
            node_target_tensor = torch.FloatTensor(node_target_seq)
            edge_target_tensor = torch.FloatTensor(edge_target_seq)
            return {
                'node_input': node_input_tensor,  # (N, history_T, 2)
                'edge_input': edge_input_tensor,  # (M, history_T, 2)
                'node_target': node_target_tensor,  # (N, future_T, 1) - level
                'edge_target': edge_target_tensor  # (M, future_T, 1) - flow
            }
def create_time_series_dataloader_train(level_df, inflow_df, flow_df, velocity_df, history_T, future_T,
                                  batch_size=32, stride=1, shuffle=True, num_workers=0, cache=True):
    """
    Create time-series DataLoader

    Args:
    level_df: DataFrame of shape (total_time, N), node depth data
    inflow_df: DataFrame of shape (total_time, N), node inflow data
    flow_df: DataFrame of shape (total_time, M), edge flow data
    velocity_df: DataFrame of shape (total_time, M), edge velocity data
    history_T: history window length
    future_T: forecast horizon
    batch_size: batch size
    stride: sliding window stride
    shuffle: Whether to shuffle
    num_workers: DataLoader num_workers (0 recommended with cache)
    cache: Cache all samples in memory (default True, speeds up training)

    Returns:
    DataLoader instance
    """
    # Create dataset
    dataset = TimeSeriesGraphDataset(
        level_df=level_df,
        inflow_df=inflow_df,
        flow_df=flow_df,
        velocity_df=velocity_df,
        history_T=history_T,
        future_T=future_T,
        stride=stride,
        cache=cache
    )

    # With cache, set num_workers=0 to avoid multiprocessing overhead
    if cache and num_workers > 0:
        print(f"Warning: with cache, num_workers=0 is recommended. Current num_workers={num_workers}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if not cache else 0,  # Use single process when cache is enabled
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader

def create_time_series_dataloader_test(level_df, target_level_df, inflow_df, flow_df, target_flow_df, velocity_df, target_velocity_df,
                                       history_T, future_T, batch_size=32, stride=1, shuffle=True, num_workers=0, cache=True):
    """
    Create time-series DataLoader

    Args:
    level_df: DataFrame of shape (total_time, N), node depth data
    inflow_df: DataFrame of shape (total_time, N), node inflow data
    flow_df: DataFrame of shape (total_time, M), edge flow data
    velocity_df: DataFrame of shape (total_time, M), edge velocity data
    history_T: history window length
    future_T: forecast horizon
    batch_size: batch size
    stride: sliding window stride
    shuffle: Whether to shuffle
    num_workers: DataLoader num_workers (0 recommended with cache)
    cache: Cache all samples in memory (default True, speeds up training)

    Returns:
    DataLoader instance
    """
    # Create dataset
    dataset = TimeSeriesGraphDataset_Test(
        level_df=level_df,
        target_level_df=target_level_df,
        inflow_df=inflow_df,
        flow_df=flow_df,
        target_flow_df=target_flow_df,
        velocity_df=velocity_df,
        target_velocity_df=target_velocity_df,
        history_T=history_T,
        future_T=future_T,
        stride=stride,
        cache=cache
    )

    # With cache, set num_workers=0 to avoid multiprocessing overhead
    if cache and num_workers > 0:
        print(f"Warning: with cache, num_workers=0 is recommended. Current num_workers={num_workers}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if not cache else 0,  # Use single process when cache is enabled
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader

# Hard-coded conversion (works when there are exactly 162 key-value pairs)
def dict_to_tensor_fixed(data):
    sorted_keys = sorted(data.keys())

    first_values = []
    second_values = []

    for key in sorted_keys:
        value = data[key]
        if isinstance(value, (tuple, list)) and len(value) == 2:
            first, second = value
            first_values.append(first)
            second_values.append(second)
        else:
            first_values.append(0)
            second_values.append(0)

    tensor_data = torch.tensor([first_values, second_values], dtype=torch.long)
    return tensor_data

def dataset_create(graph_data_path='', node_inflow_path='',
                   train_node_depth_path='',
                   train_link_flow_path='',
                   train_link_velocity_path='',
                   test_node_depth_path='',
                   test_link_flow_path='',
                   test_link_velocity_path='',
                   start_time='2025-04-01 00:00:00', end_time='2025-05-01 00:00:00',
                   use_cache=True, force_recreate=False, train_batch_size=32, test_batch_size=32, future_T=1, history_T=1):
    """
    Create time-series graph dataset and DataLoaders
    
    Args:
    graph_data_path: User-provided graph archive path
    node_depth_path: Path to node depth CSV
    node_inflow_path: Path to node inflow CSV
    link_flow_path: Path to edge flow CSV
    link_velocity_path: Path to edge velocity CSV
    use_cache: Whether to use pkl cache (default True)
    force_recreate: Force recreate DataLoaders and ignore cache (default False)
    
    Returns:
    train_data_loader, test_data_loader: Train and test DataLoaders
    """
    # Load graph archive
    graph_data = np.load(graph_data_path, allow_pickle=True)
    node_dict = graph_data['node_dict']
    node_pos = graph_data['node_pos']
    link_dict = graph_data['link_dict']

    edge_index = graph_data['edge_index'].item()

    edge_index = dict_to_tensor_fixed(edge_index)

    node_type_arr = graph_data['node_type_arr']
    link_type_arr = graph_data['link_type_arr']
    node_elev_arr = graph_data['node_elev_arr']
    node_elev_list = []
    for node_elev in node_elev_arr:
        node_elev_list.append(node_elev[0])
    node_max_depth_arr = graph_data['node_max_depth_arr']
    node_area_arr = graph_data['node_area_arr']
    node_area_list = []
    for node_area in node_area_arr:
        node_area_list.append(node_area[0])
    link_length_arr = graph_data['link_length_arr']
    link_length_arr = link_length_arr.astype(int)
    link_length_list = []
    for link_length in link_length_arr:
        link_length_list.append(link_length[0])

    link_diameter_arr = graph_data['link_diameter_arr']
    link_diameter_list = []
    for link_diameter in link_diameter_arr:
        link_diameter_list.append(link_diameter[0])
    link_roughness_arr = graph_data['link_roughness_arr']
    link_roughness_list = []
    for link_roughness in link_roughness_arr:
        link_roughness_list.append(link_roughness[0])
    link_offset_up_arr = graph_data['link_offset_up_arr']
    link_offset_up_list = []
    for link_offset_up in link_offset_up_arr:
        link_offset_up_list.append(link_offset_up[0])
    link_offset_down_arr = graph_data['link_offset_down_arr']
    link_offset_down_list = []
    for link_offset_down in link_offset_down_arr:
        link_offset_down_list.append(link_offset_down[0])
    # Configure sensor/mask asset IDs from your private graph_data export
    sensor_nodes = []
    sensor_links = []
    mask_nodes = []
    mask_links = []
    # Find dict keys for sensor nodes and pipe segments
    # If node_dict is ndarray, call .item() to get the dict
    if isinstance(node_dict, np.ndarray):
        node_dict = node_dict.item()
    if isinstance(link_dict, np.ndarray):
        link_dict = link_dict.item()
    
    sensor_node_indices = []
    for sensor in sensor_nodes:
        for key, value in node_dict.items():
            if value == sensor:
                sensor_node_indices.append(key)
                break
    sensor_link_indices = []
    for sensor in sensor_links:
        for key, value in link_dict.items():
            if value == sensor:
                sensor_link_indices.append(key)
                break
    mask_node_indices = []
    for mask in mask_nodes:
        for key, value in node_dict.items():
            if value == mask:
                mask_node_indices.append(key)
                break
    mask_link_indices = []
    for mask in mask_links:
        for key, value in link_dict.items():
            if value == mask:
                mask_link_indices.append(key)
                break

    # Load CSV files
    node_depth = pd.read_csv(train_node_depth_path, index_col=0, parse_dates=True, encoding='utf-8')
    target_node_depth = pd.read_csv(test_node_depth_path, index_col=0, parse_dates=True, encoding='utf-8')
    node_inflow = pd.read_csv(node_inflow_path, index_col=0, parse_dates=True, encoding='utf-8')
    link_flow = pd.read_csv(train_link_flow_path, index_col=0, parse_dates=True, encoding='utf-8')
    target_link_flow = pd.read_csv(test_link_flow_path, index_col=0, parse_dates=True, encoding='utf-8')
    link_velocity = pd.read_csv(train_link_velocity_path, index_col=0, parse_dates=True, encoding='utf-8')
    target_link_velocity = pd.read_csv(test_link_velocity_path, index_col=0, parse_dates=True, encoding='utf-8')
    shred = 1e-7
    max_depth_list = [x + shred for x in node_depth.max().tolist()]
    min_depth_list = [x + shred for x in node_depth.min().tolist()]
    max_inflow_list = [x + shred for x in node_inflow.max().tolist()]
    min_inflow_list = [x + shred for x in node_inflow.min().tolist()]
    max_flow_list = [x + shred for x in link_flow.max().tolist()]
    min_flow_list = [x + shred for x in link_flow.min().tolist()]
    max_velocity_list = [x + shred for x in link_velocity.max().tolist()]
    min_velocity_list = [x + shred for x in link_velocity.min().tolist()]
    max_lists = [max_depth_list, max_inflow_list, max_flow_list, max_velocity_list]
    min_lists = [min_depth_list, min_inflow_list, min_flow_list, min_velocity_list]
    for i in range(len(min_lists)):
        max_lists[i] = np.sign(max_lists[i]) * 1e-7 + max_lists[i]
        min_lists[i] = np.sign(min_lists[i]) * 1e-7 + min_lists[i]
    # Map node_depth columns to dict keys
    # Map node_inflow columns to dict keys
    node_new_columns = []
    for col in node_inflow.columns:
        for key, value in node_dict.items():
            if value == col:
                node_new_columns.append(key)
                break
    node_inflow.columns = node_new_columns
    # Reorder all inputs by numeric column index
    node_depth = node_depth.reindex(sorted(node_depth.columns), axis=1)
    target_node_depth = target_node_depth.reindex(sorted(target_node_depth.columns), axis=1)
    node_inflow = node_inflow.reindex(sorted(node_inflow.columns), axis=1)
    link_flow = link_flow.reindex(sorted(link_flow.columns), axis=1)
    target_link_flow = target_link_flow.reindex(sorted(target_link_flow.columns), axis=1)
    link_velocity = link_velocity.reindex(sorted(link_velocity.columns), axis=1)
    target_link_velocity = target_link_velocity.reindex(sorted(target_link_velocity.columns), axis=1)
    # Filter data by time range
    node_depth = node_depth.loc[start_time:end_time]
    target_node_depth = target_node_depth.loc[start_time:end_time]
    node_inflow = node_inflow.loc[start_time:end_time]
    link_flow = link_flow.loc[start_time:end_time]
    target_link_flow = target_link_flow.loc[start_time:end_time]
    link_velocity = link_velocity.loc[start_time:end_time]
    target_link_velocity = target_link_velocity.loc[start_time:end_time]
    # Split train and test sets
    split_ratio = 0.8
    split_index = int(len(node_depth) * split_ratio)
    node_depth_train = node_depth.iloc[:split_index]
    node_depth_test = node_depth.iloc[split_index:]
    node_depth_target = target_node_depth.iloc[split_index:]
    node_inflow_train = node_inflow.iloc[:split_index]
    node_inflow_test = node_inflow.iloc[split_index:]
    link_flow_train = link_flow.iloc[:split_index]
    link_flow_test = link_flow.iloc[split_index:]
    link_flow_target = target_link_flow.iloc[split_index:]
    link_velocity_train = link_velocity.iloc[:split_index]
    link_velocity_test = link_velocity.iloc[split_index:]
    link_velocity_target = target_link_velocity.iloc[split_index:]
    
    # Define cache file paths
    cache_dir = ''
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cache_dir, 'train_dataloader.pkl')
    test_cache_path = os.path.join(cache_dir, 'test_dataloader.pkl')
    
    train_data_loader = None
    test_data_loader = None
    
    # If cache enabled and not forced, try loading cache
    if use_cache and not force_recreate:
        # Try loading train DataLoader cache
        if os.path.exists(train_cache_path):
            print("Loading train DataLoader from cache...")
            try:
                with open(train_cache_path, 'rb') as f:
                    train_data_loader = pickle.load(f)
                print("Train DataLoader loaded successfully!")
            except Exception as e:
                print(f"Failed to load train DataLoader cache: {e}, recreating...")
                train_data_loader = None
        
        # Try loading test DataLoader cache
        if os.path.exists(test_cache_path):
            print("Loading test DataLoader from cache...")
            try:
                with open(test_cache_path, 'rb') as f:
                    test_data_loader = pickle.load(f)
                print("Test DataLoader loaded successfully!")
            except Exception as e:
                print(f"Failed to load test DataLoader cache: {e}, recreating...")
                test_data_loader = None
    
    # If cache missing or load failed, create new DataLoaders
    if train_data_loader is None:
        print("Creating train DataLoader...")
        train_data_loader = create_time_series_dataloader_train(
            level_df=node_depth_train,
            inflow_df=node_inflow_train,
            flow_df=link_flow_train,
            velocity_df=link_velocity_train,
            history_T=history_T,
            future_T=future_T,
            batch_size=train_batch_size,
            stride=1,
            shuffle=True,
            num_workers=0,  # Use 0 workers in cache mode for best performance
            cache=True
        )
        # If cache enabled, save train DataLoader to pkl
        if use_cache:
            print("Saving train DataLoader to cache...")
            with open(train_cache_path, 'wb') as f:
                pickle.dump(train_data_loader, f)
            print("Train DataLoader saved successfully!")
    
    if test_data_loader is None:
        print("Creating test DataLoader...")
        test_data_loader = create_time_series_dataloader_test(
            level_df=node_depth_test,
            target_level_df=node_depth_target,
            inflow_df=node_inflow_test,
            flow_df=link_flow_test,
            target_flow_df=link_flow_target,
            velocity_df=link_velocity_test,
            target_velocity_df=link_velocity_target,
            history_T=history_T,
            future_T=future_T,
            batch_size=test_batch_size,
            stride=future_T,
            shuffle=False,
            num_workers=0,  # Use 0 workers in cache mode for best performance
            cache=True
        )
        # If cache enabled, save test DataLoader to pkl
        if use_cache:
            print("Saving test DataLoader to cache...")
            with open(test_cache_path, 'wb') as f:
                pickle.dump(test_data_loader, f)
            print("Test DataLoader saved successfully!")
    
    print("DataLoaders ready.")
    return (edge_index, train_data_loader, test_data_loader, sensor_node_indices, sensor_link_indices, mask_node_indices, mask_link_indices,
            link_length_list, link_diameter_list, link_roughness_list, link_offset_up_list, link_offset_down_list, node_elev_list,
            node_area_list, node_pos, max_lists, min_lists)

if __name__ == '__main__':
    (edge_index, train_data_loader, test_data_loader, sensor_node_indices, sensor_link_indices, mask_node_indices, mask_link_indices,
     link_length_arr, link_diameter_arr, link_roughness_arr, link_offset_up_arr, link_offset_down_arr, node_elev_arr,
     node_area_arr, node_pos, max_lists, min_lists) = dataset_create(force_recreate=True, train_batch_size=8, test_batch_size=8)
    # Test DataLoader output shapes
    for batch in train_data_loader:
        print("Train Batch - Node Input Shape:", batch['node_input'].shape)
        print("Train Batch - Edge Input Shape:", batch['edge_input'].shape)
        print("Train Batch - Node Target Shape:", batch['node_target'].shape)
        print("Train Batch - Edge Target Shape:", batch['edge_target'].shape)
        break
    for batch in test_data_loader:
        print("Test Batch - Node Input Shape:", batch['node_input'].shape)
        print("Test Batch - Edge Input Shape:", batch['edge_input'].shape)
        print("Test Batch - Node Target Shape:", batch['node_target'].shape)
        print("Test Batch - Edge Target Shape:", batch['edge_target'].shape)
        break
import numpy as np
import h5py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from pynwb import NWBHDF5IO


def load_monkey_I(file_path, filename, batch_size=256, window_size=20, fold = 0):
    """
    Loader data from Monkey I.
    
    Args:
        file_path: Path to the data directory
        filename: Name of the file to load 
        batch_size: Batch size for the data loaders
        window_size: Number of time steps to include in each input window
        fold: which fold of 5 fold cross validation are we on
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
    """
    # Ensure filename has .mat extension
    filename = filename if filename.endswith(".mat") else filename + ".mat"
    full_path = os.path.join(file_path, filename)
    
    assert os.path.exists(full_path), f"File {full_path} does not exist"
    
    # Load the data
    print(f"Loading {filename}")
    dataset = h5py.File(full_path, "r")
    
    # Extract data
    spikes = dataset["spikes"][()]
    cursor_pos = dataset["cursor_pos"][()]
    t = np.squeeze(dataset["t"][()])
    sampling_rate = 4e-3
    
  
    input_feature_size = 96


    # Create time bins
    new_t = np.arange(t[0] - 0.004, t[-1], sampling_rate)
    
    # Initialize 3D array for spikes
    spike_train = np.zeros((*spikes.shape, len(new_t)), dtype=np.int8)
    
    # Fill in spikes
    for row_idx, row in enumerate(spikes):
        for col_idx, element in enumerate(row):
            if isinstance(element, np.ndarray):
                bins, _ = np.histogram(element, bins=new_t.squeeze())
            else:
                bins, _ = np.histogram(dataset[element][()], bins=new_t.squeeze())
            idx = np.nonzero(bins)[0] + 1
            spike_train[row_idx, col_idx, idx] = 1
    
    # Combine spikes from same electrode (OR operation)
    spike_train = np.bitwise_or.reduce(spike_train, axis=0)
    # Calculate velocities from cursor positions
    velocity = np.gradient(cursor_pos, axis=1)
    # Process velocity using mark_movement function
    x_vels = velocity[0, :]
    y_vels = velocity[1, :]
    


    velocity_array = np.stack((x_vels, y_vels))
    
    # Create input-output pairs
    n_timesteps = spike_train.shape[1]
    n_sequences = n_timesteps - window_size
    
    # Initialize arrays for sequences
    X = np.zeros((n_sequences, input_feature_size))
    y = np.zeros((n_sequences, 1))
    
    speeds = np.zeros(n_sequences)
    # Create sequences
    for i in range(n_sequences):
        # Input: sum spikes over the window
        X[i] = np.sum(spike_train[:, i:i+window_size], axis=1)
        
        # Output: speed (magnitude of velocity) at the final timestep
        x_vel = velocity_array[0, i+window_size-1]
        y_vel = velocity_array[1, i+window_size-1]
        speed = np.sqrt(x_vel**2 + y_vel**2)
        y[i, 0] = speed
        speeds[i] = speed
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    k = 5
    n_sequences = len(X_tensor)
    fold_size = n_sequences // k

    fold_speeds = []
    for i in range(k): 
        fold_start = i * fold_size
        fold_end = fold_start + fold_size if i != k - 1 else n_sequences
        fold_speeds.append(speeds[fold_start:fold_end])
    #fold_speeds = np.concatenate(fold_speeds)

    # Determine the start and end indices of the validation fold
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold != k - 1 else n_sequences  # last fold gets remainder

    # Slice validation set
    test_X = X_tensor[val_start:val_end]
    test_y = y_tensor[val_start:val_end]

    # Slice training set (everything except current fold)
    train_X = torch.cat((X_tensor[:val_start], X_tensor[val_end:]), dim=0)
    train_y = torch.cat((y_tensor[:val_start], y_tensor[val_end:]), dim=0)
    # Create datasets
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


def mark_movement(input, zero_thresh = 0.05, amp_thresh = 9, min_length = 25):
    movement_array = np.zeros(len(input))
    movement_array.fill(np.nan)
    res = np.where(abs(input) > zero_thresh, input, movement_array)
    
    mask = ~np.isnan(res)
    idx = np.where(mask[1:] != mask[:-1])[0] + 1
    idx = np.concatenate(([0], idx, [len(res)]))
    
    for start, end in zip(idx[:-1], idx[1:]):
        if mask[start]:
            sequence = res[start:end]

            if np.max(np.abs(sequence)) < amp_thresh:
                res[start:end] = np.nan
                
            elif np.max(np.abs(sequence)) > 100:
                res[start:end] = np.nan
                
            #filter by duration (disregard very short movements)
            if len(sequence) < min_length:
                res[start:end] = np.nan
    return res


def load_monkey_M(file_path, file, batch_size, window_size = 20, fold = 0):
    """
    Loader data from Monkey I.
    
    Args:
        file_path: Path to the data directory
        filename: Name of the file to load 
        batch_size: Batch size for the data loaders
        window_size: Number of time steps to include in each input window
        fold: which fold of 5 fold cross validation are we on
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
    """
    m1_matrix = [[] for _ in range(96)]
    
    full_file = os.path.join(file_path, file)
    assert os.path.exists(full_file), f"File {full_file} does not exist"
    print(f"Loading {file}")

    
    with NWBHDF5IO(full_file, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units
        units_df = units.to_dataframe()
        
        behavior_module = nwbfile.processing['behavior']
        velocity = behavior_module.data_interfaces['Velocity']
        cursor_vel = velocity['cursor_vel']
        vel_array = cursor_vel.data[:]
        start_time = cursor_vel.timestamps[0]
        end_time = cursor_vel.timestamps[-1]
        
        # Process velocity data
        x_vels = vel_array[:,0]
        y_vels = vel_array[:,1]
        marked_x = mark_movement(x_vels)
        marked_y = mark_movement(y_vels)
        smoothed_x = np.nan_to_num(marked_x, nan=0)
        smoothed_y = np.nan_to_num(marked_y, nan=0)
        velocity_array = np.stack((smoothed_x, smoothed_y))
        
        # Extract M1 spike times
        for index, row in units_df.iterrows():
            electrode_id = row['electrodes'].index.item()
            if electrode_id < 96:  # Only process M1 electrodes (0-95)
                m1_matrix[electrode_id].append(row['spike_times'])
    
    # Flatten spike times for each electrode
    m1_flattened_matrix = []
    for row in m1_matrix:
        if len(row) > 0:
            combined = np.concatenate(row)
            m1_flattened_matrix.append(np.sort(combined))
        else:
            m1_flattened_matrix.append([])
    
    # Calculate number of time bins
    bin_size = 0.01  # 10ms bins
    
    # if not weird_init:
    #     n_bins = int((end_time - start_time) / bin_size) + 1
    # else: 
    #     n_bins = int((end_time - start_time) / bin_size) + 2
    n_bins = int((end_time - start_time) / bin_size) + 1
    
    # Initialize binary matrix
    m1_binary_spikes = np.zeros((96, n_bins))
    
    # Fill in spikes 
    for neuron_idx, spike_times in enumerate(m1_flattened_matrix):
        if len(spike_times) > 0:
            # Convert times to indices
            time_indices = ((spike_times - start_time) / bin_size).astype(int)
            valid_indices = time_indices[time_indices < n_bins]
            m1_binary_spikes[neuron_idx, valid_indices] = 1
    
    m1_spike_array = m1_binary_spikes
    
    # Create bins and setup for input-output pairs
    n_timesteps = m1_spike_array.shape[1]
    
    # Calculate how many complete sequences we can make
    n_sequences = n_timesteps - window_size
    
    # Create input-output pairs where:
    # - Input: summed window of M1 activity (96 neurons summed over window)
    # - Output: speed (magnitude of velocity) at the final timestep of the window (scalar)
    
    # Initialize arrays for sequences
    X = np.zeros((n_sequences, 96))  # (sequences, M1 neurons summed over window)
    y = np.zeros((n_sequences, 1))  # (sequences, speed as scalar)
    
    # Create sequences
    for i in range(n_sequences):
        # Input: sum M1 activity over the window (96x1 feature vector)
        X[i] = np.sum(m1_spike_array[:, i:i+window_size], axis=1)
        
        # Output: speed (magnitude of velocity) at the final timestep
        x_vel = velocity_array[0, i+window_size-1]
        y_vel = velocity_array[1, i+window_size-1]
        speed = np.sqrt(x_vel**2 + y_vel**2)  # Calculate speed as magnitude of velocity
        y[i, 0] = speed
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    #print(f'Input tensor shape: {X_tensor.shape}, Output tensor shape: {y_tensor.shape}')
    k = 5
    n_sequences = len(X_tensor)
    fold_size = n_sequences // k

    # Determine the start and end indices of the validation fold
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold != k - 1 else n_sequences  # last fold gets remainder

    # Slice validation set
    test_X = X_tensor[val_start:val_end]
    test_y = y_tensor[val_start:val_end]

    # Slice training set (everything except current fold)
    train_X = torch.cat((X_tensor[:val_start], X_tensor[val_end:]), dim=0)
    train_y = torch.cat((y_tensor[:val_start], y_tensor[val_end:]), dim=0)
    
    
    # Create datasets
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
    return train_loader, test_loader
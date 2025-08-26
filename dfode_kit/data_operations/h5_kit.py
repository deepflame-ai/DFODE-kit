import h5py
import torch
import numpy as np
import cantera as ct

from dfode_kit.utils import BCT, inverse_BCT

def touch_h5(hdf5_file_path):
    """
    Load an HDF5 file and print its contents and metadata.

    Parameters
    ----------
    hdf5_file_path : str
        The path to the HDF5 file to be opened.

    Returns
    -------
    None
        This function does not return any value. It prints the metadata, groups,
        and datasets contained in the HDF5 file.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    OSError
        If the file cannot be opened as an HDF5 file.

    Notes
    -----
    This function provides a simple way to inspect the structure and metadata of 
    an HDF5 file, making it useful for debugging and understanding data organization.

    Examples
    --------
    >>> touch_h5('/path/to/file.h5')
    Metadata in the HDF5 file:
    root_directory: /path/to/root
    mechanism: /path/to/mechanism.yaml
    species_names: ['T', 'p', 'species1', ...]

    Groups and datasets in the HDF5 file:
    Group: scalar_fields
      Dataset: 0, Shape: (100, 5)
      Dataset: 1, Shape: (100, 5)
    Group: mesh
      Dataset: Cx, Shape: (100, 1)
    """
    print(f"Inspecting HDF5 file: {hdf5_file_path}\n")
    
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Print the metadata
        print("Metadata in the HDF5 file:")
        for attr in hdf5_file.attrs:
            print(f"{attr}: {hdf5_file.attrs[attr]}")
        
        # Print the names of the groups and datasets in the file
        print("\nGroups and datasets in the HDF5 file:")
        for group_name, group in hdf5_file.items():
            print(f"Group: {group_name}")
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                print(f"  Dataset: {dataset_name}, Shape: {dataset.shape}")

def get_TPY_from_h5(file_path):
    """
    Reads the scalar_fields group from an HDF5 file and stacks its datasets into a single array.

    Parameters
    ----------
    file_path : str
        The path to the HDF5 file containing the 'scalar_fields' group.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array containing the stacked datasets from the 'scalar_fields' group.

    Raises
    ------
    KeyError
        If the 'scalar_fields' group does not exist in the HDF5 file.
    OSError
        If the file cannot be opened or read.

    Notes
    -----
    This function retrieves all datasets within the 'scalar_fields' group of the 
    specified HDF5 file and stacks them vertically into a single 2D array, where 
    each dataset corresponds to a row in the resulting array.

    Examples
    --------
    >>> stacked_data = get_TPY_from_h5('/path/to/file.h5')
    >>> print(stacked_data.shape)
    (num_datasets, num_columns)
    """
    with h5py.File(file_path, 'r') as f:
        # Access the 'scalar_fields' group
        scalar_fields_group = f['scalar_fields']
        
        # Get the number of datasets in the scalar_fields group
        dataset_count = len(scalar_fields_group)
        print(f'Number of datasets in scalar_fields group: {dataset_count}')
        
        # Read and stack all datasets into one large array
        data_list = []
        
        for dataset_name in scalar_fields_group:
            dataset = scalar_fields_group[dataset_name][:]
            data_list.append(dataset)
        
        # Stack all datasets vertically (along a new axis)
        stacked_data = np.vstack(data_list)
    
    return stacked_data

def advance_reactor(gas, state, reactor, reactor_net, time_step):
    """Advance the reactor simulation for a given state."""
    state = state.flatten()
    
    expected_shape = (2 + gas.n_species,)
    assert state.shape == expected_shape
    
    gas.TPY = state[0], state[1], state[2:]
    
    reactor.syncState()
    reactor_net.reinitialize()
    reactor_net.advance(time_step)
    reactor_net.set_initial_time(0.0)
    
    return gas

@torch.no_grad()
def load_model(model_path, device, model_class, model_layers):
    state_dict = torch.load(model_path, map_location='cpu')
    
    model = model_class(model_layers)
    model.load_state_dict(state_dict['net'])
    
    model.eval()
    model.to(device=device)
    
    return model

@torch.no_grad()
def predict_Y(model, model_path, d_arr, mech, device):
    gas = ct.Solution(mech)
    n_species = gas.n_species
    expected_dims = 2 + n_species
    assert d_arr.shape[1] == expected_dims
    
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    Xmu0 = state_dict['data_in_mean']
    Xstd0 = state_dict['data_in_std']
    Ymu0 = state_dict['data_target_mean']
    Ystd0 = state_dict['data_target_std']
    
    d_arr = np.clip(d_arr, 0, None)
    d_arr[:, 1] *= 0
    d_arr[:, 1] += 101325
    
    orig_Y = d_arr[:, 2:].copy()
    in_bct = d_arr.copy()
    in_bct[:, 2:] = BCT(in_bct[:, 2:])
    in_bct_norm = (in_bct - Xmu0) / Xstd0      
    
    input = torch.from_numpy(in_bct_norm).float().to(device=device)
    
    output = model(input)
    
    out_bct = output.cpu().numpy() * Ystd0 + Ymu0 + in_bct[:, 2:-1]
    next_Y = orig_Y.copy()
    next_Y[:, :-1] = inverse_BCT(out_bct)
    next_Y[:, :-1] = next_Y[:, :-1] / np.sum(next_Y[:, :-1], axis=1, keepdims=True) * (1 - next_Y[:, -1:])
    
    return next_Y

@torch.no_grad()
def nn_integrate(orig_arr, model_path, device, model_class, model_layers, time_step, mech, frozen_temperature=305):
    model = load_model(model_path, device, model_class, model_layers)
    
    mask = orig_arr[:, 0] > frozen_temperature
    infer_arr = orig_arr[mask, :]
    
    next_Y = predict_Y(model, model_path, infer_arr, mech, device)
    
    new_states = np.hstack((np.zeros((orig_arr.shape[0], 1)), orig_arr))
    new_states[:, 0] += time_step
    new_states[:, 2] = orig_arr[:, 1]
    new_states[mask, 3:] = next_Y
    
    setter_gas = ct.Solution(mech)
    getter_gas = ct.Solution(mech)
    new_T = np.zeros_like(next_Y[:, 0])
    
    for idx, (state, next_y) in enumerate(zip(infer_arr, next_Y)):
        try:
            setter_gas.TPY = state[0], state[1], state[2:]
            h = setter_gas.enthalpy_mass
            
            getter_gas.Y = next_y
            getter_gas.HP = h, state[1]
            
            new_T[idx] = getter_gas.T
        
        except ct.CanteraError as e:
            continue  # Skip this iteration or set a default value
    new_states[mask, 1] = new_T
    
    return new_states

def integrate_h5(
    file_path, 
    time_step, 
    cvode_integration=True,
    nn_integration=False,
    model_settings=None,
):
    """
    Process datasets from an HDF5 file, applying CVODE or neural network integration,
    and save the results in corresponding groups within the file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing scalar fields.
    time_step : float
        Time step for the reactor simulation or neural network inference.
    cvode_integration : bool, optional
        If True, processes data using CVODE integration and saves it in the 
        `cvode_integration` group. Default is True.
    nn_integration : bool, optional
        If True, processes data using a neural network integration and saves it in 
        the `nn_integration` group. Default is False.
    model_settings : dict, optional
        A dictionary containing model settings for the neural network integration. 
        Must include keys: 'model_path', 'device', 'model_class', 'model_layers', 
        'time_step', and 'mech'.

    Returns
    -------
    None
    """
    data_dict = {}
    
    with h5py.File(file_path, 'r') as f:
        mech = f.attrs['mechanism']
        scalar_fields_group = f['scalar_fields']
        
        for dataset_name in scalar_fields_group:
            dataset = scalar_fields_group[dataset_name][:]
            data_dict[dataset_name] = dataset  # Store in a dictionary
    
    if cvode_integration:
        gas = ct.Solution(mech)
        reactor = ct.Reactor(gas, name='Reactor1', energy='off')
        reactor_net = ct.ReactorNet([reactor])
        reactor_net.rtol, reactor_net.atol = 1e-6, 1e-10
        
        processed_data_dict = {}
        
        for name, data in data_dict.items():
            processed_data = np.empty((data.shape[0], data.shape[1]+1))
            for i, state in enumerate(data):
                gas = advance_reactor(gas, state, reactor, reactor_net, time_step)
                
                new_state = np.array([time_step, gas.T, gas.P] + list(gas.Y))
                
                processed_data[i, :] = new_state
            
            print(processed_data[0])
            print(processed_data[-1])
            processed_data_dict[name] = processed_data
        
        with h5py.File(file_path, 'a') as f:  # Use 'a' to append
            cvode_group = f.create_group('cvode_integration')
            
            for dataset_name, processed_data in processed_data_dict.items():
                cvode_group.create_dataset(dataset_name, data=processed_data)
                print(f'Saved processed dataset: {dataset_name} in cvode_integration group')

    if nn_integration:
        processed_data_dict = {}
        if model_settings is None:
            raise ValueError("model_settings must be provided for neural network integration.")
        
        for name, data in data_dict.items():
            try:
                processed_data = nn_integrate(data, **model_settings)
                processed_data_dict[name] = processed_data
            except Exception as e:
                print(f"Error processing dataset '{name}': {e}")
            
        with h5py.File(file_path, 'a') as f:  # Use 'a' to append
            if 'nn_integration' in f:
                del f['nn_integration']  # Delete the existing group
            nn_group = f.create_group('nn_integration')
            
            for dataset_name, processed_data in processed_data_dict.items():
                nn_group.create_dataset(dataset_name, data=processed_data)
                print(f'Saved processed dataset: {dataset_name} in nn_integration group')

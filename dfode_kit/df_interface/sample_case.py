from pathlib import Path

import h5py
import numpy as np
import cantera as ct

from dfode_kit.utils import is_number, read_openfoam_scalar

def concatenate_species_arrays(species_names, directory_path):
    """Concatenate scalar arrays from OpenFOAM files for each species in the specified directory."""
    all_arrays = []
    directory_path = Path(directory_path)
    
    # Ensure the provided directory exists
    if not Path(directory_path).is_dir():
        raise ValueError(f"The directory does not exist: {directory_path}")
    
    for species in species_names:
        # Construct the file name based on species name
        file_path = directory_path / species
        
        # Check if the file exists
        if file_path.is_file():
            try:
                # Read the scalar values using the existing function
                species_array = read_openfoam_scalar(file_path)
                all_arrays.append(species_array)
            except ValueError as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    # Concatenate all arrays into one big array
    if all_arrays:
        concatenated_array = np.concatenate(all_arrays, axis=1)
        return concatenated_array
    else:
        raise ValueError("No valid species arrays found to concatenate.")

def save_arrays_to_hdf5(root_dir, mechanism, hdf5_file_path):
    """Iterate through directories in root_dir, concatenate arrays, and save to HDF5."""
    root_path = Path(root_dir).resolve()
    mechanism = Path(mechanism).resolve()
    hdf5_file_path = Path(hdf5_file_path)
    gas = ct.Solution(mechanism)
    species_names = ['T', 'p'] + gas.species_names
    print(f"Species names: {species_names}")
    
    
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        hdf5_file.attrs['root_directory'] = str(root_path)
        hdf5_file.attrs['mechanism'] = str(mechanism)
        hdf5_file.attrs['species_names'] = species_names
        
        
        numeric_dirs = [
            dir_path for dir_path in root_path.iterdir()
            if dir_path.is_dir() and is_number(dir_path.name) and dir_path.name != '0'
        ]
        
        # Sort the directories based on their numeric values
        numeric_dirs.sort(key=lambda x: float(x.name))
        
        for dir_path in numeric_dirs:
            # Concatenate arrays for the current directory
            try:
                concatenated_array = concatenate_species_arrays(species_names, dir_path)
                
                # Create a dataset in HDF5 with the directory path as the key
                hdf5_file.create_dataset(str(dir_path.name), data=concatenated_array)
            except ValueError as e:
                print(f"Error processing directory {dir_path}: {e}")

    print(f"Saved concatenated arrays to {hdf5_file_path}")
    

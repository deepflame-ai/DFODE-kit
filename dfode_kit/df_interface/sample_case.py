from pathlib import Path

import h5py
import numpy as np
import cantera as ct

from dfode_kit.utils import is_number, read_openfoam_scalar

def gather_species_arrays(species_names, directory_path) -> np.ndarray:
    """
    Concatenate scalar arrays from OpenFOAM files for each species in the specified directory.

    Parameters
    ----------
    species_names : list of str
        A list of species names corresponding to the files in the directory.
    directory_path : str
        The path to the directory containing the OpenFOAM files.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array containing the concatenated scalar arrays for each species.
    
    Raises
    ------
    ValueError
        If the provided directory does not exist, if there is a shape mismatch 
        between arrays, or if no valid species arrays are found to concatenate.

    Notes
    -----
    This function reads scalar values from files named after the species and 
    ensures that all arrays have the same number of cells. If a scalar value 
    is a float, it is converted into a uniform numpy array.

    Examples
    --------
    >>> gather_species_arrays(['species1', 'species2'], '/path/to/directory')
    array([[...], [...], ...])
    """
    all_arrays = []
    num_cell = None
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
                
                # Check if species_array is a numpy array
                if isinstance(species_array, np.ndarray):
                    # Assign num_cell on the first valid species_array
                    if num_cell is None:
                        num_cell = species_array.shape[0]
                    else:
                        # Ensure the shape matches num_cell
                        if species_array.shape[0] != num_cell:
                            raise ValueError(f"Shape mismatch for {species}: expected {num_cell}, got {species_array.shape[0]}.")

                all_arrays.append(species_array)
            except ValueError as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    # Replace non-numpy arrays (that are floats) with numpy arrays of uniform values
    for i in range(len(all_arrays)):
        if not isinstance(all_arrays[i], np.ndarray):
            if isinstance(all_arrays[i], float):
                all_arrays[i] = np.full((num_cell, 1), all_arrays[i])  # Create a uniform array
            else:
                print(f"Warning: {all_arrays[i]} is not a numpy array or float.")

    # Concatenate all arrays into one big array if there are valid arrays
    if all_arrays:
        concatenated_array = np.concatenate(all_arrays, axis=1)
        return concatenated_array
    else:
        raise ValueError("No valid species arrays found to concatenate.")

def df_to_h5(root_dir, mechanism, hdf5_file_path, include_mesh=True):
    """
    Iterate through directories in root_dir, concatenate arrays, and save to an HDF5 file.

    Parameters
    ----------
    root_dir : str
        The path to the root directory containing subdirectories with data.
    mechanism : str
        The path to the mechanism file to be used by the Cantera solution.
    hdf5_file_path : str
        The path where the HDF5 file will be saved.
    include_mesh : bool, optional
        Whether to include mesh data in the HDF5 file (default is True).

    Returns
    -------
    None
        This function does not return any value. It saves the concatenated data 
        directly to an HDF5 file.

    Raises
    ------
    ValueError
        If there are issues with reading directories or files, or if the data 
        cannot be processed.

    Notes
    -----
    This function processes directories containing numerical data, concatenates 
    scalar arrays for each species, and saves the results in an HDF5 file. 
    It also optionally includes mesh data from predefined mesh files.

    Examples
    --------
    >>> df_to_h5('/path/to/root', '/path/to/mechanism.yaml', '/path/to/output.h5')
    Saved concatenated arrays to /path/to/output.h5
    """
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
        
        scalar_group = hdf5_file.create_group('scalar_fields')
        
        numeric_dirs = [
            dir_path for dir_path in root_path.iterdir()
            if dir_path.is_dir() and is_number(dir_path.name) and dir_path.name != '0'
        ]
        
        # Sort the directories based on their numeric values
        numeric_dirs.sort(key=lambda x: float(x.name))
        
        for dir_path in numeric_dirs:
            # Concatenate arrays for the current directory
            try:
                concatenated_array = gather_species_arrays(species_names, dir_path)
                
                # Create a dataset in HDF5 with the directory path as the key
                scalar_group.create_dataset(str(dir_path.name), data=concatenated_array)
            except ValueError as e:
                print(f"Error processing directory {dir_path}: {e}")
        
        if include_mesh:
            mesh_group = hdf5_file.create_group('mesh')
            mesh_files = [
                root_path / 'temp/0/Cx',
                root_path / 'temp/0/Cy',
                root_path / 'temp/0/Cz',
                root_path / 'temp/0/V',
            ]
            
            for mesh_file in mesh_files:
                if mesh_file.is_file():
                    try:
                        mesh_data = read_openfoam_scalar(mesh_file)
                        mesh_group.create_dataset(str(mesh_file.name), data=mesh_data)
                    except ValueError as e:
                        print(f"Error reading mesh file {mesh_file}: {e}")
                else:
                    print(f"Mesh file not found: {mesh_file}")

    print(f"Saved concatenated arrays to {hdf5_file_path}")
    

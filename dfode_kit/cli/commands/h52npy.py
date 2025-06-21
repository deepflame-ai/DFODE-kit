import argparse
import h5py
import numpy as np

def add_command_parser(subparsers):
    h52npy_parser = subparsers.add_parser('h52npy', help='Convert HDF5 scalar fields to NumPy array.')
    h52npy_parser.add_argument('--source', 
                               required=True,
                               type=str, 
                               help='Path to the HDF5 file.')
    h52npy_parser.add_argument('--save_to', 
                               required=True,
                               type=str, 
                               help='Path for the output NumPy file.')

def handle_command(args):
    print("Handling h52npy command")
    # Load the HDF5 file and concatenate datasets
    concatenate_datasets_to_npy(args.source, args.save_to)

def concatenate_datasets_to_npy(hdf5_file_path, output_npy_file):
    """Concatenate all datasets under the 'scalar_fields' group and save to a NumPy file."""
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Check if the 'scalar_fields' group exists
        if 'scalar_fields' not in hdf5_file:
            raise ValueError(f"'scalar_fields' group not found in {hdf5_file_path}")

        scalar_group = hdf5_file['scalar_fields']
        all_arrays = []

        # Iterate through all datasets in the 'scalar_fields' group
        for dataset_name in scalar_group.keys():
            dataset = scalar_group[dataset_name]
            all_arrays.append(dataset[:])  # Read the dataset and append to list

        # Print the number of datasets
        num_datasets = len(all_arrays)
        print(f"Number of datasets in 'scalar_fields': {num_datasets}")

        # Concatenate all arrays along the first axis
        concatenated_array = np.concatenate(all_arrays, axis=0)

        # Print the shape of the final concatenated array
        print(f"Shape of the final concatenated array: {concatenated_array.shape}")

        # Save the concatenated array to a .npy file
        np.save(output_npy_file, concatenated_array)
        print(f"Saved concatenated array to {output_npy_file}")

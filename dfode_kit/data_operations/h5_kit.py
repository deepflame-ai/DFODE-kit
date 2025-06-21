import h5py

def load_and_print_hdf5_contents(hdf5_file_path):
    """Load an HDF5 file and print its contents and metadata."""
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
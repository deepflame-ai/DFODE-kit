import h5py

def load_and_print_hdf5_contents(hdf5_file_path):
    """Load an HDF5 file and print its contents and metadata."""
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Print the metadata
        print("Metadata in the HDF5 file:")
        for attr in hdf5_file.attrs:
            print(f"{attr}: {hdf5_file.attrs[attr]}")
        
        # Print the names of the datasets in the file
        print("\nDatasets in the HDF5 file:")
        for dataset_name in hdf5_file.keys():
            print(f"Dataset: {dataset_name}")
            # Print the contents of each dataset
            data = hdf5_file[dataset_name][:]
            print(data)
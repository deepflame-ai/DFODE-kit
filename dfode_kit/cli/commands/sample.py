import argparse
from dfode_kit.df_interface.sample_case import df_to_h5
from dfode_kit.data_operations.h5_kit import touch_h5

def add_command_parser(subparsers):
    sample_parser = subparsers.add_parser('sample', help='Perform sampling.')
    
    # Add arguments for the sample command
    sample_parser.add_argument(
        '--mech', 
        required=True,
        type=str, 
        help='Path to the mechanism file.'
    )
    sample_parser.add_argument(
        '--case', 
        required=True,
        type=str, 
        help='Root directory containing data.'
    )
    sample_parser.add_argument(
        '--save', 
        required=True,
        type=str, 
        help='Path where the HDF5 file will be saved.'
    )
    sample_parser.add_argument(
        '--include_mesh', 
        action='store_true', 
        help='Include mesh data in the HDF5 file.'
    )

def handle_command(args):
    print("Handling sample command")
    # Call the save_arrays_to_hdf5 function with the parsed arguments
    df_to_h5(args.case, args.mech, args.save, include_mesh=args.include_mesh)
    print()
    
    # Optionally load and print the contents of the HDF5 file
    touch_h5(args.save)
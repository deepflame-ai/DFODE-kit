import numpy as np
from dfode_kit.data_operations.augment_data import random_perturb
from dfode_kit.data_operations.h5_kit import get_TPY_from_h5

def add_command_parser(subparsers):
    augment_parser = subparsers.add_parser('augment', help='Perform data augmentation.')
    
    # Add specific arguments for the augment command here
    augment_parser.add_argument(
        '--mech', 
        required=True,
        type=str, 
        help='Path to the YAML mechanism file.'
    )
    augment_parser.add_argument(
        '--h5_file', 
        required=True,
        type=str,
        help='Path to the h5 file to augment.'
    )
    augment_parser.add_argument(
        '--output_file',
        required=True,
        type=str,
        help='Path to the output NUMPY file.' 
    )
    augment_parser.add_argument(
        '--heat_limit',
        type=bool,
        default=False,
        help='contraint perturbed data with heat release.'
    )
    augment_parser.add_argument(
        '--element_limit',
        type=bool,
        default=True,
        help='contraint perturbed data with element ratio.'
    )
    augment_parser.add_argument(
        '--dataset_num',
        required=True,
        type=int,
        help='num of dataset.'
    )
    augment_parser.add_argument(
        '--perturb_factor',
        type=float,
        default=0.1,
        help='Factor to perturb the data by.'
    )

def handle_command(args):
    print("Handling augment command")
    
    print(f"Loading data from h5 file: {args.h5_file}")
    data = get_TPY_from_h5(args.h5_file)
    print("Data shape:", data.shape)


    All_data = random_perturb(data, args.mech, args.dataset_num, args.heat_limit, args.element_limit, args.perturb_factor)

    np.save(args.output_file, All_data)
    print("Saved augmented data shape:", All_data.shape)
    print(f"Saved augmented data to {args.output_file}")
import numpy as np
from dfode_kit.data_operations.augment_data import random_perturb
from dfode_kit.data_operations.h5_kit import get_TPY_from_h5

def add_command_parser(subparsers):
    augment_parser = subparsers.add_parser('augment', help='Perform data augmentation.')
    
    # Add specific arguments for the augment command here
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
        help='Path to the output h5 file.'
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
    aug_data = random_perturb(data, alpha=args.perturb_factor)
    
    np.save(args.output_file, aug_data)
    print(f"Saved augmented data to {args.output_file}")
import argparse
import numpy as np
from dfode_kit.data_operations import label_npy as label_main

def add_command_parser(subparsers):
    label_parser = subparsers.add_parser('label', help='Label data.')
    
    label_parser.add_argument(
        '--mech', 
        required=True,
        type=str, 
        help='Path to the YAML mechanism file.'
    )
    label_parser.add_argument(
        '--time', 
        required=True,
        type=float, 
        help='Time step for reactor advancement'
    )
    label_parser.add_argument('--source', 
        required=True,
        type=str, 
        help='Path to the original dataset.'
    )
    label_parser.add_argument('--save',
        required=True, 
        type=str, 
        help='Path to save the labeled dataset.'
    )
    label_parser.set_defaults(func=handle_command)

def handle_command(args):
    try:
        labeled_data = label_main(
            mech_path=args.mech,
            time_step=float(args.time),
            source_path=args.source
        )
        np.save(args.save, labeled_data)
        print(f"Labeled data saved to: {args.save}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
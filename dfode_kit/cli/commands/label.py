import argparse
from dfode_kit.data_operations.label_data import main as label_main

def parse_time_steps(time_steps_str):
    """Parse time steps from a string formatted as '(min max)'."""
    try:
        # Remove parentheses and split by space
        min_step, max_step = map(float, time_steps_str.strip('()').split())
        return min_step, max_step
    except ValueError:
        raise argparse.ArgumentTypeError("Time steps must be provided in the format (min max).")

def add_command_parser(subparsers):
    label_parser = subparsers.add_parser('label', help='Label data.')
    
    label_parser.add_argument('--mech', 
                              required=True,
                              type=str, 
                              help='Path to the YAML mechanism file.')
    label_parser.add_argument('--time', 
                              required=True,
                              type=parse_time_steps, 
                              help='Time range for reactor advancement in the format \'(min max)\'.')
    label_parser.add_argument('--source', 
                              required=True,
                              type=str, 
                              help='Path to the original dataset.')
    label_parser.add_argument('--save',
                              required=True, 
                              type=str, 
                              help='Path to save the labeled dataset.')
    label_parser.set_defaults(func=handle_command)

def handle_command(args):
    try:
        min_time_step, max_time_step = args.time_steps
        label_main(args.original_data_path, args.mech, min_time_step, max_time_step, args.save_file_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
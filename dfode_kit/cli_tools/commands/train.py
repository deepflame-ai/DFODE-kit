from dfode_kit.dfode_core.train.train import train

def add_command_parser(subparsers):
    train_parser = subparsers.add_parser('train', help='Train the model.')
    train_parser.add_argument(
        '--mech', 
        required=True,
        type=str, 
        help='Path to the YAML mechanism file.'
    )
    train_parser.add_argument(
        '--source_file',
        required=True,
        type=str,
        help='Path to the source NUMPY file.' 
    )
    train_parser.add_argument(
        '--output_path',
        required=True,
        type=str,
        help='Path to the output model.' 
    )
    # Add specific arguments for the train command here

def handle_command(args):
    print("Handling train command")
    # Implement your train logic here

    train(args.mech,args.source_file,args.output_path)

    print(f"Saved Model to {args.output_path}")
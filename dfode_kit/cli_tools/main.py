import argparse
from dfode_kit.cli_tools.command_loader import load_commands

def main():
    parser = argparse.ArgumentParser(prog='dfode-kit', description=(
    'dfode-kit provides a command-line interface for performing various tasks '
    'related to deep learning and reacting flow simulations. This toolkit allows '
    'users to efficiently augment data, label datasets, sample from low-dimensional '
    'flame simulations, and train deep learning models. It is designed to support '
    'physics-informed methodologies for accurate and reliable simulations.'
))
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Load and register main commands
    commands = load_commands()
    for command_name, command_module in commands.items():
        command_module.add_command_parser(subparsers)

    args = parser.parse_args()

    # Call the appropriate command handler based on the command
    if args.command in commands:
        commands[args.command].handle_command(args)
    else:
        print(f"Command '{args.command}' not found.")

if __name__ == "__main__":
    main()
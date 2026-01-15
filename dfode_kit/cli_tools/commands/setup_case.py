import argparse
import json
import yaml
from pathlib import Path
from dfode_kit.agent_interface import DFODEAgentInterface

def add_command_parser(subparsers):
    parser = subparsers.add_parser('setup_case', help='Initialize and configure a new simulation case.')
    
    parser.add_argument(
        '--work_dir', 
        required=True,
        type=str, 
        help='Directory to create/setup the case in.'
    )
    parser.add_argument(
        '--config', 
        required=True,
        type=str, 
        help='Path to a JSON or YAML configuration file.'
    )
    parser.add_argument(
        '--template', 
        default='oneD_freely_propagating_flame',
        type=str, 
        help='Name of the template to use (default: oneD_freely_propagating_flame).'
    )

def handle_command(args):
    print(f"Setting up case in {args.work_dir}...")
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
            
    agent = DFODEAgentInterface()
    
    try:
        # Step 1: Create Workspace
        agent.create_workspace(args.work_dir, args.template)
        
        # Step 2: Configure
        agent.setup_simulation(args.work_dir, config_dict, template_name=args.template)
        
        print(f"Success! Case is ready at {args.work_dir}.")
        print("You can now run the simulation using 'dfode-kit run_sim ...'")
        
    except Exception as e:
        print(f"Error during setup: {e}")

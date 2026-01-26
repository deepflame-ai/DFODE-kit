import argparse
from dfode_kit.agent_interface import DFODEAgentInterface

def add_command_parser(subparsers):
    parser = subparsers.add_parser('run_sim', help='Run the simulation (e.g., ./Allrun).')
    
    parser.add_argument(
        '--work_dir', 
        required=True,
        type=str, 
        help='Directory where the case is set up.'
    )
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=None,
        help='Maximum time in seconds to wait for the simulation.'
    )

def handle_command(args):
    print(f"Running simulation in {args.work_dir}...")
    
    agent = DFODEAgentInterface()
    
    try:
        agent.run_simulation(args.work_dir, timeout=args.timeout)
        print(f"Simulation finished successfully in {args.work_dir}.")
    except Exception as e:
        print(f"Simulation failed: {e}")

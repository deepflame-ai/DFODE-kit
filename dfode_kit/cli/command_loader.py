import importlib
import pkgutil

def load_commands(package_name='dfode_kit.cli.commands'):
    commands = {}

    # Dynamically load modules from the specified package
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        module = importlib.import_module(module_name)
        
        # Check if the module has the required functions
        if hasattr(module, 'add_command_parser') and hasattr(module, 'handle_command'):
            commands[module_name.split('.')[-1]] = module

    return commands
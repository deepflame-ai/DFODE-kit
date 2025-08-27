from setuptools import setup, find_packages

setup(
    name='dfode',
    version='0.1',
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        'numpy',
        'cantera',
    ],
    entry_points={
        'console_scripts': [
            'dfode-kit = dfode_kit.cli_tools.main:main',  # Main entry point for CLI
            'new_subcommand = dfode_kit.new_subcommand:handle_command',
        ],
    },
)
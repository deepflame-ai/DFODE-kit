from setuptools import setup, find_packages

setup(
    name='dfode',
    version='0.1',
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        'numpy',
        'cantera<3.1',
        'h5py',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'dfode-kit = dfode_kit.cli_tools.main:main',  # Main entry point for CLI
        ],
    },
)
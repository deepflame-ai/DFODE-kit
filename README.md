# DFODE-kit: Deep Learning Package for Combustion Kinetics

DFODE-kit is an open-source Python package designed to accelerate combustion simulations by efficiently solving flame chemical kinetics governed by high-dimensional stiff ordinary differential equations (ODEs). This package integrates deep learning methodologies to replace conventional numerical integration, enabling significant speedups and improved accuracy.

## Features
- **Efficient Sampling Module**: Extracts high-quality thermochemical states from low-dimensional manifolds in canonical flames.
- **Data Augmentation**: Enhances training datasets to approximate high-dimensional composition spaces in turbulent flames.
- **Neural Network Implementation**: Supports optimized training with physical constraints to ensure model fidelity.
- **Seamless Integration**: Easily deploy trained models within the DeepFlame CFD solver or other platforms like OpenFOAM.
- **Robust Performance**: Achieves high accuracy with up to two orders of magnitude speedup in various combustion scenarios.

## Installation
To install DFODE-kit, clone the repository and install the dependencies:

```bash
git clone https://github.com/DeepFlame-ML/DFODE-kit.git
cd DFODE
pip install -e .
```

## Usage
Once you have installed DFODE-kit, you can use it to sample data, augment datasets, train models, and make predictions. Below is a basic command-line interface (CLI) format:

```bash
dfode-kit CMD ARGS
```


### Commands Available:
- `sample`: Perform raw data sampling from canonical flame simulations.
- `augment`: Apply random noise and physical constraints to improve the training dataset.
- `label`: Generate supervised learning labels using Cantera's CVODE solver.
- `train`: Train neural network models based on the specified datasets and parameters.

A comprehensive tutorial guide of [tutorial](https://github.com/DeepFlame-ML/DFODE-kit/blob/main/tutorials/oneD_freely_propagating_flame/readme.md) is provided to help you get started quickly. Besides, the Jupyter Notebook version [notebook](https://github.com/DeepFlame-ML/DFODE-kit/blob/main/tutorials/oneD_freely_propagating_flame/dfode_kit_tutorial.ipynb) is provided here.


## Directories
- **dfode-kit**: Main procedure and functions.
- **mechanisms**: Thermochemical mechanism folder.
- **model_test**: Model testing with priori and posteriori validataions
- **tutorials**: Tutorials with sampling cases 


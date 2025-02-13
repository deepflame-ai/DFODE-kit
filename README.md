# DFODE: Deep Learning Framework for ODEs in Combustion

DFODE is an open-source deep learning package designed to accelerate computationally intensive reacting flow simulations by replacing conventional numerical integration of chemical source terms governed by high-dimensional stiff ordinary differential equations (ODEs).

## Overview

DFODE provides:
- Efficient sampling module for extracting high-quality base data from low-dimensional manifolds in canonical flames
- Robust data augmentation strategy to expand training data for high-dimensional turbulent flames
- Neural network implementation with optimized data preprocessing and hyperparameter tuning
- Seamless interfaces for deploying trained models within the open-source CFD solver [DeepFlame](https://github.com/deepmodeling/deepflame-dev)
- Physical constraints derived from conservation laws to ensure reliability in CFD applications

## Features

- Low-dimensional manifold sampling from canonical flame configurations
  - 0D homogeneous reactors
  - 1D laminar premixed flames  
  - 2D counterflow diffusion flames
  - 1D detonation tubes

- Data augmentation with physics-based constraints
  - Random perturbation of thermochemical states
  - Heat release and molar element ratio filtering
  - Mass conservation enforcement

- Deep neural network model
  - Multi-layer perceptron architecture
  - Box-Cox transformation for data preprocessing
  - Physics-informed loss functions
  - Support for both CPU and GPU training/inference

- Physical-aware correction
  - Elemental conservation enforcement
  - Energy conservation constraints
  - Heat release prediction error control

- CFD solver integration
  - Seamless interface with DeepFlame
  - Support for both CPU and GPU inference
  - Flexible deployment options

## Installation

```bash
# Clone the repository
git clone https://github.com/DeepFlame-ML/DFODE.git

# Install dependencies
pip install -r requirements.txt

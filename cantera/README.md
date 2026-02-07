# Cantera 1D Freely Propagating Flame Simulation

This directory contains scripts for simulating 1D freely-propagating premixed H2/air flames using Cantera 3.2.0.

## Requirements

```bash
pip install cantera>=3.2.0 numpy matplotlib
```

Or using uv:
```bash
uv pip install cantera>=3.2.0 numpy matplotlib
```

## Usage

Run the main simulation script:

```bash
python 1d_freely_propagating_flame.py
```

After simulation, plot results:

```bash
# Plot all iterations and final solution
python plot_flame.py

# Plot a specific file (quick view)
python quick_plot.py output_data/final_flame.npz
python quick_plot.py output_data/iteration_00006.npz
```

## Data Augmentation

Augment flame data through linear interpolation:

```bash
python augment_flame.py <npz_file> [num_points]

# Example: create 500 interpolated points
python augment_flame.py output_data/iteration_00006.npz 500

# Example: create 1000 interpolated points from final solution
python augment_flame.py output_data/final_flame.npz 1000
```

The augmentation algorithm creates synthetic data by:
1. Defining an evenly spaced temperature grid from Tmin to Tmax
2. Linearly interpolating pressure and species mass fractions between adjacent original points
3. Normalizing species fractions to sum to 1

Output:
- `augmented_data/*_augmented.npz` - Augmented datasets
- `augmented_data/*_augmented_comparison.png` - Visualization comparing original vs interpolated data

## Collect and Perturb

Collect all augmented data into a single dataset and apply perturbations:

```bash
python collect_augmented.py augmented_data ../mechanisms/Burke2012_s9r23.yaml 5000 0.15 1
```

Outputs:
- `collected_data.npz` - Perturbed dataset with `T`, `p`, and `Y_{species}`
- `collected_data_unperturbed.npz` - Unperturbed collected dataset for comparison

## Visualization: Perturbed vs Final Flame Profile

Scatter plot major species vs temperature (perturbed) with the final flame profile line, plus the final flame profile layout:

```bash
python visualize_perturbation.py collected_data.npz output_data/final_flame.npz
```

Outputs:
- `plots/perturbation_species_vs_temperature.png`
- `plots/flame_final_solution.png`

## Output

The script generates:

1. **Iteration data**: NPZ files for each grid refinement saved in `output_data/`
   - Each file contains named arrays: `grid`, `velocity`, `temperature`, `density`, `pressure`, `mass_fraction_*`, `mole_fraction_*`, `thermal_conductivity`, `heat_release_rate`, `flame_speed`, `transport_model`
   - A new file is created each time Cantera inserts grid points based on refinement criteria
   - Grid typically grows from ~8 points to ~150-200 points through 10-15 refinements

2. **Final solution**: `output_data/final_flame.npz` with the converged solution

## Configuration

Edit the simulation parameters in `main()`:
- `p`: Pressure (Pa)
- `Tin`: Unburned gas temperature (K)
- `equivalence_ratio`: Equivalence ratio for H2/air (phi=1 is stoichiometric)
- `width`: Flame width (m)
- `loglevel`: Verbosity (0-8)

## Mechanism

Uses `Burke2012_s9r23.yaml` from `../mechanisms/` - a detailed hydrogen/oxygen mechanism with 9 species (H, H2, O, OH, H2O, O2, HO2, H2O2, N2).

## Example: Loading NPZ Data

```python
import numpy as np

data = np.load('output_data/final_flame.npz')
grid = data['grid']
temperature = data['temperature']
mass_fraction_H2 = data['mass_fraction_H2']
flame_speed = data['flame_speed']
```

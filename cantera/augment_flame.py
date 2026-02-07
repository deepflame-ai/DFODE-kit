"""
Data Augmentation through Linear Interpolation

Algorithm: flame-structure-interp

Input: Thermochemical dataset from 1D laminar flame simulation, sorted by spatial position in 1D domain
       (includes temperature (T), pressure (p), and species mass fractions (Y))

Input: A grid of evenly spaced temperature values within [T_min, T_max] of original dataset:
       T_grid = [T_min, T_min + ΔT, T_min + 2ΔT, ..., T_max]

For each adjacent pair (T_i, p_i, Y_i) and (T_{i+1}, p_{i+1}, Y_{i+1}) in dataset:
    For each T_new in T_grid:
        If (T_i < T_new < T_{i+1}):
            Compute interpolation for pressure (p_new):
            [
            p_new = p_i + (p_{i+1} - p_i) / (T_{i+1} - T_i) * (T_new - T_i)
            ]
            Compute interpolation for species mass fractions (Y_new):
            [
            Y_new = Y_i + (Y_{i+1} - Y_i) / (T_{i+1} - T_i) * (T_new - T_i)
            ]
            Store (T_new, p_new, Y_new)
        EndIf
EndFor

Output: Augmented dataset containing both original and interpolated thermochemical states
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def augment_flame_data(npz_file, num_points=500, output_dir="augmented_data"):
    """
    Augment flame data through linear interpolation.

    Args:
        npz_file: Path to NPZ file containing flame data
        num_points: Number of points to interpolate between Tmin and Tmax
        output_dir: Directory to save augmented data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load original data
    data = np.load(npz_file)
    orig_grid = data["grid"]
    orig_T = data["temperature"]
    orig_p = data["pressure"]
    orig_velocity = data["velocity"]
    orig_heat_release = data["heat_release_rate"]

    # Get species data (mass fractions Y)
    species_keys = sorted([k for k in data.keys() if k.startswith("Y_")])
    species_names = [k[2:] for k in species_keys]
    orig_Y = np.stack([data[f"Y_{species}"] for species in species_names], axis=1)

    print(f"Original data: {len(orig_T)} grid points")
    print(f"Temperature range: {orig_T.min():.1f} - {orig_T.max():.1f} K")
    print(f"Species: {species_names}")

    # Create temperature grid for interpolation
    T_min, T_max = orig_T.min(), orig_T.max()
    T_grid = np.linspace(T_min, T_max, num_points)
    ΔT = T_grid[1] - T_grid[0]

    print(f"\nInterpolation grid:")
    print(f"  Temperature points: {num_points}")
    print(f"  Temperature range: {T_min:.1f} - {T_max:.1f} K")
    print(f"  Temperature spacing ΔT: {ΔT:.2f} K")

    # Initialize arrays for augmented data
    aug_T = T_grid
    aug_p = np.full(num_points, orig_p)  # Pressure is constant
    aug_Y = np.zeros((num_points, len(species_names)))

    # Interpolate each species
    for i, species in enumerate(species_names):
        aug_Y[:, i] = np.interp(T_grid, orig_T, data[f"Y_{species}"])

    # Interpolate other variables
    aug_velocity = np.interp(T_grid, orig_T, orig_velocity)
    aug_heat_release = np.interp(T_grid, orig_T, orig_heat_release)
    aug_grid = np.interp(T_grid, orig_T, orig_grid)

    # Clip Y to physical range [0, 1]
    aug_Y = np.clip(aug_Y, 0, 1)

    # Normalize Y to sum to 1 (excluding minor corrections)
    Y_sum = np.sum(aug_Y, axis=1)
    mask = Y_sum > 1e-10
    aug_Y[mask] = aug_Y[mask] / Y_sum[mask, np.newaxis]

    print(f"\nAugmented data: {num_points} points")
    print(f"Y shape: {aug_Y.shape}")
    print(
        f"Y sum check: min={np.sum(aug_Y, axis=1).min():.6f}, max={np.sum(aug_Y, axis=1).max():.6f}"
    )

    # Save augmented data
    output_file = output_dir / f"{Path(npz_file).stem}_augmented.npz"

    aug_data = {
        "grid": aug_grid,
        "velocity": aug_velocity,
        "temperature": aug_T,
        "density": np.interp(T_grid, orig_T, data["density"]),
        "pressure": aug_p,
        "heat_release_rate": aug_heat_release,
        "flame_speed": data.get("flame_speed", 0.0),
    }

    # Copy optional metadata
    for key in ["iteration", "equivalence_ratio", "transport_model"]:
        if key in data.files:
            aug_data[key] = data[key]

    for i, species in enumerate(species_names):
        aug_data[f"Y_{species}"] = aug_Y[:, i]
        aug_data[f"X_{species}"] = np.interp(T_grid, orig_T, data[f"X_{species}"])

    np.savez(output_file, **aug_data)
    print(f"\nSaved augmented data to: {output_file}")

    return output_file


def visualize_augmentation(npz_file, orig_npz_file, output_dir="augmented_data"):
    """
    Visualize original vs augmented data.

    Args:
        npz_file: Path to augmented NPZ file
        orig_npz_file: Path to original NPZ file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load original and augmented data
    orig_data = np.load(orig_npz_file)
    aug_data = np.load(npz_file)

    # Get data
    orig_T = orig_data["temperature"]
    aug_T = aug_data["temperature"]

    species_keys = sorted([k for k in aug_data.keys() if k.startswith("Y_")])
    species_names = [k[2:] for k in species_keys]

    # Major species
    major_species = ["H2", "O2", "H2O", "H", "O", "OH"]
    major_species = [s for s in major_species if s in species_names]

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Augmentation Visualization\nOriginal: {len(orig_T)} points → Augmented: {len(aug_T)} points",
        fontsize=14,
    )

    # Plot species vs temperature
    for idx, species in enumerate(major_species):
        ax = axes[idx // 3, idx % 3]

        orig_Y = orig_data[f"Y_{species}"]
        aug_Y = aug_data[f"Y_{species}"]

        # Filter out very small values for cleaner plot
        mask = orig_Y > 1e-4
        if not np.any(mask):
            mask = orig_Y > 1e-6

        ax.plot(
            orig_T[mask], orig_Y[mask], "ro", markersize=3, alpha=0.6, label="Original"
        )
        ax.plot(aug_T, aug_Y, "b-", linewidth=1.5, alpha=0.8, label="Interpolated")

        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel(f"Y_{species} (Mass Fraction)")
        ax.set_title(species)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / f"{Path(npz_file).stem}_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_file}")

    plt.close()

    return output_file


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python augment_flame.py <npz_file> [num_points]")
        print("\nExample:")
        print("  python augment_flame.py output_data/iteration_00006.npz 500")
        print("  python augment_flame.py output_data/final_flame.npz 1000")
        print("\nDefault num_points: 500")
        sys.exit(1)

    npz_file = Path(sys.argv[1])
    if not npz_file.exists():
        print(f"Error: File '{npz_file}' not found.")
        sys.exit(1)

    # Get number of points
    num_points = 500
    if len(sys.argv) >= 3:
        num_points = int(sys.argv[2])

    # Augment data
    output_file = augment_flame_data(npz_file, num_points=num_points)

    # Visualize
    visualize_augmentation(output_file, npz_file)

    print("\n" + "=" * 60)
    print("Augmentation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
